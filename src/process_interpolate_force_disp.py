"""Interpolate separate tensile-test signals and calculate force-displacement data.

Each experiment directory must contain a force-time CSV with ``time_s`` and
``force_N`` columns and a distance-time CSV with ``time_s`` and
``marker_dist_m`` columns. The script validates and cleans both inputs, then
interpolates one signal onto the selected time base over their overlapping time
range. The merged data is passed to the same processing functions used by
``process_force_disp.py`` to determine L0, calculate displacement, and write the
force-displacement CSV and JSON summary. Optional debug exports preserve the
interpolated and full processed tables.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .process_force_disp import compute_force_displacement, save_outputs
except ImportError:
    from process_force_disp import compute_force_displacement, save_outputs


## Configuration constants.
# Paths.
IN_ROOT = Path("data/raw_input")
OUT_ROOT = Path("data/processed_force_disp")
FORCE_FILENAME = "force_time.csv"
DISTANCE_FILENAME = "dist_time.csv"

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
EXPORT_DEBUG = False


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

    out = df[required_columns].copy().dropna()
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna()

    if out.empty:
        raise ValueError(f"No numeric rows found in {path} after parsing {required_columns}.")

    n_rows_input = len(out)
    value_column = required_columns[1]
    rows_per_time = out.groupby(TIME_COLUMN).size()
    repeated_times = rows_per_time[rows_per_time > 1]
    out = out.sort_values(TIME_COLUMN).groupby(TIME_COLUMN, as_index=False).mean(numeric_only=True)

    averaging = {
        "n_rows_input": int(n_rows_input),
        "n_rows_after_averaging": int(len(out)),
        "n_time_points_averaged": int(len(repeated_times)),
        "n_rows_collapsed_by_averaging": int(n_rows_input - len(out)),
        "averaging_applied": bool(len(repeated_times)),
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
        force_path = exp_dir / FORCE_FILENAME
        distance_path = exp_dir / DISTANCE_FILENAME

        has_force = force_path.is_file()
        has_distance = distance_path.is_file()

        if has_force and has_distance:
            pairs[exp_dir.name] = (force_path, distance_path)
        elif has_force or has_distance:
            missing = DISTANCE_FILENAME if has_force else FORCE_FILENAME
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


# Process every valid experiment using the configuration constants above.
def main() -> None:
    if not IN_ROOT.exists() or not IN_ROOT.is_dir():
        raise SystemExit(f"Raw input folder not found: {IN_ROOT}")
        
    try:
        experiment_pairs = _index_experiment_pairs(IN_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not experiment_pairs:
        raise SystemExit(
            f"No valid experiment folders found under {IN_ROOT}. "
            f"Expected {FORCE_FILENAME} and {DISTANCE_FILENAME} in each experiment folder."
        )

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
