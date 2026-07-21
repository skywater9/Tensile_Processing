"""Interpolate separate tensile-test signals and calculate force-displacement data.

Each experiment directory must contain ``<experiment>_force_time.csv`` with
``time_s`` and ``force_N`` columns and ``<experiment>_dist_time.csv`` with
``time_s`` and one or more ``marker_dist_m`` columns. Multiple marker-distance
columns are averaged row-by-row. The script validates and cleans both inputs, then
interpolates one signal onto the selected time base over their overlapping time
range. The script then determines L0, calculates displacement, and writes the
force-displacement CSV and JSON summary. Optional debug exports preserve the
interpolated and full processed tables.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


## Configuration constants.
# Paths.
IN_ROOT = Path("data/raw_input")
OUT_ROOT = Path("data/processed_force_disp")
FORCE_FILENAME_SUFFIX = "_force_time.csv"
DISTANCE_FILENAME_SUFFIX = "_dist_time.csv"

# Required input column names.
TIME_COLUMN = "time_s"
FORCE_COLUMN = "force_N"
DISTANCE_COLUMN = "marker_dist_m"
INPUT_COLUMNS = {
    "force": [TIME_COLUMN, FORCE_COLUMN],
    "distance": [TIME_COLUMN, DISTANCE_COLUMN],
}

# Processing parameters.
TIME_BASE = "force"  # Use "force" or "distance".
F_THRESH_MIN_N = 1.0
F_THRESH_FRAC_OF_MAX = 0.01
FORCE_CUTOFF_N = 25  # Set to 0 to keep the complete force-displacement curve.
EXPORT_DEBUG = False
CSV_FLOAT_FORMAT = "%.12f"


def _estimate_l0_from_initial_segment(
    marker_dist_m: np.ndarray,
    force_n: np.ndarray,
    f_thresh_n: float,
    min_points: int = 3,
    fallback_points: int = 10,
) -> float:
    if len(marker_dist_m) == 0:
        return float("nan")

    valid = np.isfinite(marker_dist_m) & np.isfinite(force_n)
    if not valid.any():
        return float("nan")

    low_force = valid & (force_n <= f_thresh_n)

    initial_count = 0
    for is_low in low_force:
        if is_low:
            initial_count += 1
        else:
            break

    if initial_count >= min_points:
        return float(np.nanmedian(marker_dist_m[:initial_count]))

    valid_marker = marker_dist_m[valid]
    n_fallback = min(fallback_points, len(valid_marker))
    return float(np.nanmedian(valid_marker[:n_fallback]))


def compute_force_displacement(
    preprocessing: pd.DataFrame,
    f_thresh_min_N: float = 1.0,
    f_thresh_frac_of_max: float = 0.01,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    time_s = preprocessing[TIME_COLUMN].to_numpy(float)
    marker_dist_m = preprocessing[DISTANCE_COLUMN].to_numpy(float)
    force_n = preprocessing[FORCE_COLUMN].to_numpy(float)

    f_max = float(np.nanmax(force_n)) if len(force_n) else float("nan")
    f_thresh = float(max(f_thresh_min_N, f_thresh_frac_of_max * f_max))

    l0_m = _estimate_l0_from_initial_segment(marker_dist_m, force_n, f_thresh)

    displacement_m = marker_dist_m - l0_m
    full = pd.DataFrame(
        {
            TIME_COLUMN: time_s,
            DISTANCE_COLUMN: marker_dist_m,
            FORCE_COLUMN: force_n,
            "displacement_m": displacement_m,
        }
    )
    final = full[["displacement_m", FORCE_COLUMN]].copy().dropna()

    summary = {
        "processing": {
            "F_thresh_N": f_thresh,
            "L0_m": l0_m,
            "n_rows_preprocessing": int(len(preprocessing)),
            "n_rows_output": int(len(final)),
        },
        "metrics": {
            "max_force_N": float(np.nanmax(force_n)) if len(force_n) else float("nan"),
            "max_displacement_m": (
                float(np.nanmax(displacement_m)) if len(displacement_m) else float("nan")
            ),
        },
    }
    return final, summary, full


def save_outputs(
    out_root: Path,
    stem: str,
    preprocessing: pd.DataFrame,
    full: pd.DataFrame,
    final: pd.DataFrame,
    summary: dict,
    export_debug: bool,
) -> None:
    out_dir = out_root / stem
    if out_dir.exists():
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    if export_debug:
        preprocessing.to_csv(
            out_dir / f"{stem}_preprocessing_export.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
        full.to_csv(
            out_dir / f"{stem}_processed_full.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
    final.to_csv(
        out_dir / f"{stem}_force_disp.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))


# Read and validate one force-time or distance-time CSV for interpolation.
def _read_time_value_table(path: Path, value_kind: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file type for {path.name}: {path.suffix}. Only .csv is supported.")

    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    if value_kind not in INPUT_COLUMNS:
        raise ValueError(f"Unsupported value kind: {value_kind}")

    required_columns = INPUT_COLUMNS[value_kind]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}. Found: {list(df.columns)}")

    value_column = required_columns[1]
    value_columns = [value_column]
    if value_kind == "distance":
        # read_csv disambiguates repeated headers as marker_dist_m.1,
        # marker_dist_m.2, etc. Treat all of them as independent readings.
        value_columns = [
            column
            for column in df.columns
            if column == value_column
            or (
                column.startswith(f"{value_column}.")
                and column.removeprefix(f"{value_column}.").isdigit()
            )
        ]

    numeric_time = pd.to_numeric(df[TIME_COLUMN], errors="coerce")
    numeric_values = df[value_columns].apply(pd.to_numeric, errors="coerce")
    valid_values_per_row = numeric_values.notna().sum(axis=1)
    n_rows_averaged_across_columns = int(
        (numeric_time.notna() & (valid_values_per_row > 1)).sum()
    )
    out = pd.DataFrame(
        {
            TIME_COLUMN: numeric_time,
            value_column: numeric_values.mean(axis=1),
        }
    )
    out = out.dropna()

    if out.empty:
        raise ValueError(f"No numeric rows found in {path} after parsing {required_columns}.")

    n_rows_input = len(out)
    rows_per_time = out.groupby(TIME_COLUMN).size()
    repeated_times = rows_per_time[rows_per_time > 1]
    out = out.sort_values(TIME_COLUMN).groupby(TIME_COLUMN, as_index=False).mean(numeric_only=True)

    averaging = {
        "n_rows_input": int(n_rows_input),
        "n_rows_after_averaging": int(len(out)),
        "n_time_points_averaged": int(len(repeated_times)),
        "n_rows_collapsed_by_averaging": int(n_rows_input - len(out)),
        "n_value_columns_input": int(len(value_columns)),
        "value_columns_input": value_columns,
        "n_rows_averaged_across_columns": n_rows_averaged_across_columns,
        "column_averaging_applied": bool(n_rows_averaged_across_columns),
        "time_averaging_applied": bool(len(repeated_times)),
        "averaging_applied": bool(n_rows_averaged_across_columns or len(repeated_times)),
    }

    if len(out) < 2:
        raise ValueError(f"Need at least 2 distinct time points in {path} for interpolation.")

    return out[TIME_COLUMN].to_numpy(float), out[value_column].to_numpy(float), averaging


# Find experiment directories containing both required input CSV files.
def _index_experiment_pairs(
    raw_root: Path,
) -> dict[str, tuple[Path, Path]]:
    pairs: dict[str, tuple[Path, Path]] = {}
    partial_missing: list[str] = []

    for exp_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        force_filename = f"{exp_dir.name}{FORCE_FILENAME_SUFFIX}"
        distance_filename = f"{exp_dir.name}{DISTANCE_FILENAME_SUFFIX}"
        force_path = exp_dir / force_filename
        distance_path = exp_dir / distance_filename

        has_force = force_path.is_file()
        has_distance = distance_path.is_file()

        if has_force and has_distance:
            pairs[exp_dir.name] = (force_path, distance_path)
        elif has_force or has_distance:
            missing = distance_filename if has_force else force_filename
            partial_missing.append(f"{exp_dir.name} (missing {missing})")

    if partial_missing:
        preview = ", ".join(partial_missing[:5])
        raise ValueError(
            "Found experiment folder(s) with only one required input file: "
            f"{preview}{' ...' if len(partial_missing) > 5 else ''}"
        )

    return pairs


# Linearly interpolate a source signal onto the selected time axis.
def _interp_on_base(base_t: np.ndarray, src_t: np.ndarray, src_v: np.ndarray) -> np.ndarray:
    return np.interp(base_t, src_t, src_v, left=np.nan, right=np.nan)


# Merge one force/distance pair over its overlapping time range.
def _combine_pair(force_path: Path, distance_path: Path) -> tuple[pd.DataFrame, dict]:
    f_t, f_v, force_averaging = _read_time_value_table(force_path, "force")
    d_t, d_v, distance_averaging = _read_time_value_table(distance_path, "distance")

    if TIME_BASE == "force":
        time_s = f_t
        force_n = f_v
        marker_dist_m = _interp_on_base(time_s, d_t, d_v)
    elif TIME_BASE == "distance":
        time_s = d_t
        marker_dist_m = d_v
        force_n = _interp_on_base(time_s, f_t, f_v)
    else:
        raise ValueError(f"Unsupported TIME_BASE: {TIME_BASE}. Use 'force' or 'distance'.")

    out = pd.DataFrame(
        {
            TIME_COLUMN: time_s,
            DISTANCE_COLUMN: marker_dist_m,
            FORCE_COLUMN: force_n,
        }
    ).dropna()

    if out.empty:
        raise ValueError(
            f"No overlapping time range between '{force_path.name}' and '{distance_path.name}' after interpolation."
        )

    interpolation = {
        "time_base": TIME_BASE,
        "n_rows_force_input": force_averaging["n_rows_input"],
        "n_rows_distance_input": distance_averaging["n_rows_input"],
        "distance_averaging": distance_averaging,
        "n_rows_interpolated": int(len(out)),
        "start_time_s": float(out[TIME_COLUMN].min()),
        "end_time_s": float(out[TIME_COLUMN].max()),
    }
    return out, interpolation


def _apply_force_cutoff(final: pd.DataFrame, force_cutoff_n: float) -> tuple[pd.DataFrame, dict]:
    if not np.isfinite(force_cutoff_n) or force_cutoff_n < 0:
        raise ValueError("Force cutoff must be a finite, non-negative value in newtons.")

    n_rows_before = len(final)
    if n_rows_before == 0:
        raise ValueError("Cannot apply a force cutoff to an empty force-displacement table.")

    force_n = final[FORCE_COLUMN].to_numpy(float)
    if force_cutoff_n == 0:
        cutoff = {
            "force_cutoff_N": 0.0,
            "force_cutoff_applied": False,
            "n_rows_before_force_cutoff": int(n_rows_before),
            "n_rows_after_force_cutoff": int(n_rows_before),
            "n_rows_removed_before_force_cutoff": 0,
            "n_rows_removed_after_force_cutoff": 0,
        }
        return final.reset_index(drop=True), cutoff

    above_cutoff = np.isfinite(force_n) & (force_n >= force_cutoff_n)
    if not above_cutoff.any():
        max_force_n = float(np.nanmax(force_n))
        raise ValueError(
            f"Force cutoff {force_cutoff_n:g} N is above the maximum recorded force "
            f"({max_force_n:g} N)."
        )

    # Keep the continuous above-cutoff region containing peak force. This avoids
    # selecting an isolated noisy crossing before loading or after fracture.
    peak_position = int(np.nanargmax(force_n))
    start_position = peak_position
    while start_position > 0 and above_cutoff[start_position - 1]:
        start_position -= 1

    end_position = peak_position
    while end_position + 1 < n_rows_before and above_cutoff[end_position + 1]:
        end_position += 1

    trimmed = final.iloc[start_position : end_position + 1].reset_index(drop=True)
    cutoff = {
        "force_cutoff_N": float(force_cutoff_n),
        "force_cutoff_applied": bool(start_position > 0 or end_position < n_rows_before - 1),
        "n_rows_before_force_cutoff": int(n_rows_before),
        "n_rows_after_force_cutoff": int(len(trimmed)),
        "n_rows_removed_before_force_cutoff": int(start_position),
        "n_rows_removed_after_force_cutoff": int(n_rows_before - end_position - 1),
    }
    return trimmed, cutoff


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate force/dist signals and compute force-displacement outputs. "
            "Provide zero, one, or many experiment folder names under data/raw_input."
        )
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Optional experiment folder names to process. If omitted, all valid folders are processed.",
    )
    parser.add_argument(
        "--force-cutoff-n",
        type=float,
        default=FORCE_CUTOFF_N,
        help=(
            "Keep the continuous force-displacement region containing peak force "
            f"at or above this force (default: {FORCE_CUTOFF_N:g} N; use 0 to disable)."
        ),
    )
    return parser.parse_args()


def _filter_experiment_pairs(
    experiment_pairs: dict[str, tuple[Path, Path]],
    selected_experiments: list[str],
) -> dict[str, tuple[Path, Path]]:
    if not selected_experiments:
        return experiment_pairs

    missing = [name for name in selected_experiments if name not in experiment_pairs]
    if missing:
        available = ", ".join(sorted(experiment_pairs))
        raise SystemExit(
            "Requested experiment folder(s) not found or incomplete: "
            f"{', '.join(missing)}. Available complete folders: {available}"
        )

    return {name: experiment_pairs[name] for name in selected_experiments}


# Process every valid experiment using the configuration constants above.
def main() -> None:
    args = _parse_args()

    if not IN_ROOT.exists() or not IN_ROOT.is_dir():
        raise SystemExit(f"Raw input folder not found: {IN_ROOT}")
        
    try:
        experiment_pairs = _index_experiment_pairs(IN_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not experiment_pairs:
        raise SystemExit(
            f"No valid experiment folders found under {IN_ROOT}. "
            "Expected <experiment>_force_time.csv and "
            "<experiment>_dist_time.csv in each experiment folder."
        )

    experiment_pairs = _filter_experiment_pairs(experiment_pairs, args.experiments)

    processed = 0
    generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    code_version = "unknown"
    try:
        code_version = f"git:{subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()}"
    except Exception:
        pass

    for experiment_name, (force_path, distance_path) in experiment_pairs.items():
        preprocessing, interpolation = _combine_pair(force_path, distance_path)
        final, summary, full = compute_force_displacement(
            preprocessing,
            f_thresh_min_N=F_THRESH_MIN_N,
            f_thresh_frac_of_max=F_THRESH_FRAC_OF_MAX,
        )
        try:
            final, force_cutoff = _apply_force_cutoff(final, args.force_cutoff_n)
        except ValueError as exc:
            raise SystemExit(f"{experiment_name}: {exc}") from exc

        summary["processing"].update(force_cutoff)
        summary["processing"]["n_rows_output"] = int(len(final))
        summary["processing"]["force_range_N"] = [
            float(final[FORCE_COLUMN].min()),
            float(final[FORCE_COLUMN].max()),
        ]
        summary["processing"]["displacement_range_m"] = [
            float(final["displacement_m"].min()),
            float(final["displacement_m"].max()),
        ]
        summary["metrics"]["max_force_N"] = float(final[FORCE_COLUMN].max())
        summary["metrics"]["max_displacement_m"] = float(final["displacement_m"].max())
        summary_out = {
            "test_group": IN_ROOT.name,
            "experiment": experiment_name,
            "generated_at_utc": generated_at_utc,
            "code_version": code_version,
            "interpolation": interpolation,
            **summary,
        }

        save_outputs(
            OUT_ROOT,
            experiment_name,
            preprocessing,
            full,
            final,
            summary_out,
            export_debug=EXPORT_DEBUG,
        )

        processed += 1
        out_path = OUT_ROOT / experiment_name / f"{experiment_name}_force_disp.csv"
        print(f"Processed: {experiment_name} ({force_path.name} + {distance_path.name}) -> {out_path}")

    print(f"Done. Wrote {processed} processed force-displacement file(s) under {OUT_ROOT}")


if __name__ == "__main__":
    main()
