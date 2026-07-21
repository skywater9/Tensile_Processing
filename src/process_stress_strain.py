"""Convert processed force-displacement data into engineering stress-strain results.

Edit the configuration constants below, then run this file directly. Each
experiment folder under INPUT_PATH must contain <experiment>_force_disp.csv and
<experiment>_summary.json. The script reads displacement and force, reuses L0
from the force-displacement summary, computes engineering stress-strain, and
writes processed CSV and JSON outputs.
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

# Configuration: edit these constants before running the script.
INPUT_PATH = Path("data/processed_force_disp")  # One combined CSV or a folder of combined CSV files.
OUT_ROOT = Path("data/processed_stress_strain")
WIDTH_MM = 10.0
THICKNESS_MM = 3.0
EXPORT_DEBUG = False
CSV_FLOAT_FORMAT = "%.12f"

FORCE_COLUMN = "force_N"


# Fit a straight line and return its slope, intercept, and R-squared value.
def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return float(m), float(b), float(r2)


# Find the best linear early-loading window for apparent Young's modulus.
def _find_apparent_modulus_window(
    strain: np.ndarray,
    stress: np.ndarray,
    *,
    min_points: int = 8,
    min_strain_span: float = 8e-4,
    stress_low_frac: float = 0.10,
    stress_high_frac: float = 0.45,
) -> dict:
    finite = np.isfinite(strain) & np.isfinite(stress)
    strain = strain[finite]
    stress = stress[finite]

    if len(strain) < min_points:
        return {
            "apparent_Youngs_modulus_GPa": float("nan"),
            "apparent_Youngs_modulus_window_MPa": [float("nan"), float("nan")],
        }

    uts = float(np.nanmax(stress))
    if not np.isfinite(uts) or uts <= 0:
        return {
            "apparent_Youngs_modulus_GPa": float("nan"),
            "apparent_Youngs_modulus_window_MPa": [float("nan"), float("nan")],
        }

    s_lo = stress_low_frac * uts
    s_hi = stress_high_frac * uts
    idx = np.flatnonzero((stress >= s_lo) & (stress <= s_hi))
    if len(idx) < min_points:
        return {
            "apparent_Youngs_modulus_GPa": float("nan"),
            "apparent_Youngs_modulus_window_MPa": [s_lo, s_hi],
        }

    strain_c = strain[idx]
    stress_c = stress[idx]

    best = None  # (score, m, r2, smin, smax)
    n = len(strain_c)

    for i in range(0, n - min_points + 1):
        for j in range(i + min_points, n + 1):
            x = strain_c[i:j]
            y = stress_c[i:j]
            span = float(x.max() - x.min())
            if span < min_strain_span:
                continue
            m, _, r2 = _linear_fit(x, y)
            if not np.isfinite(m) or m <= 0 or not np.isfinite(r2):
                continue

            score = (r2, span, (j - i))
            if best is None or score > best[0]:
                best = (score, m, r2, float(y.min()), float(y.max()))

    if best is None:
        return {
            "apparent_Youngs_modulus_GPa": float("nan"),
            "apparent_Youngs_modulus_window_MPa": [s_lo, s_hi],
        }

    _, m_best, _, s_min, s_max = best
    return {
        "apparent_Youngs_modulus_GPa": float(m_best / 1000.0),
        "apparent_Youngs_modulus_window_MPa": [s_min, s_max],
    }


# Calculate UTS, strain, toughness, and apparent modulus metrics.
def _compute_metrics(full: pd.DataFrame) -> dict:
    valid = full[np.isfinite(full["strain_eng"]) & np.isfinite(full["stress_eng_MPa"])].copy()

    if len(valid) < 5:
        return {
            "UTS_MPa": float("nan"),
            "strain_at_UTS": float("nan"),
            "max_strain_eng": float("nan"),
            "stress_range_MPa": [float("nan"), float("nan")],
            "strain_range_eng": [float("nan"), float("nan")],
            "toughness_MJ_per_m3": float("nan"),
            "apparent_Youngs_modulus_GPa": float("nan"),
            "apparent_Youngs_modulus_window_MPa": [float("nan"), float("nan")],
        }

    valid = valid.sort_values("strain_eng")
    strain = valid["strain_eng"].to_numpy(float)
    stress = valid["stress_eng_MPa"].to_numpy(float)

    uts = float(np.nanmax(stress))
    i_uts = int(np.nanargmax(stress))
    strain_at_uts = float(strain[i_uts])
    max_strain = float(np.nanmax(strain))

    toughness = float(np.trapezoid(stress, strain)) if len(strain) >= 2 else float("nan")

    modulus_info = _find_apparent_modulus_window(strain, stress)

    return {
        "UTS_MPa": uts,
        "strain_at_UTS": strain_at_uts,
        "max_strain_eng": max_strain,
        "stress_range_MPa": [float(np.nanmin(stress)), float(np.nanmax(stress))],
        "strain_range_eng": [float(np.nanmin(strain)), float(np.nanmax(strain))],
        "toughness_MJ_per_m3": toughness,
        **modulus_info,
    }


def compute_engineering_from_force_disp(
    force_disp: pd.DataFrame,
    l0_m: float,
    width_mm: float,
    thickness_mm: float,
    f_thresh_n: float,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    area_m2 = (width_mm / 1000.0) * (thickness_mm / 1000.0)

    displacement_m = force_disp["displacement_m"].to_numpy(float)
    force_n = force_disp["force_N"].to_numpy(float)

    strain_eng = displacement_m / l0_m
    stress_mpa = (force_n / area_m2) / 1e6

    full = pd.DataFrame(
        {
            "displacement_m": displacement_m,
            "force_N": force_n,
            "strain_eng": strain_eng,
            "stress_eng_MPa": stress_mpa,
        }
    )
    final = full[["strain_eng", "stress_eng_MPa"]].copy().dropna()

    metrics = _compute_metrics(full)
    summary = {
        "specimen_geometry": {
            "width_mm": width_mm,
            "thickness_mm": thickness_mm,
            "area_m2": area_m2,
        },
        "processing": {
            "F_thresh_N": f_thresh_n,
            "L0_m": l0_m,
            "n_rows_preprocessing": int(len(force_disp)),
            "n_rows_output": int(len(final)),
        },
        "metrics": metrics,
    }
    return final, summary, full


# Write final, summary, and optional debug files for one test.
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
        out_dir / f"{stem}_stress_strain.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )

    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))


# Resolve INPUT_PATH into experiment folders with force_disp + summary files.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert tensile data to engineering stress-strain outputs. "
            "Provide zero, one, or many experiment folder names under INPUT_PATH "
            "(data/processed_force_disp by default)."
        )
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Optional experiment names to process. If omitted, all valid inputs are processed.",
    )
    return parser.parse_args()


def _filter_experiment_inputs(
    experiment_inputs: dict[str, dict],
    selected_experiments: list[str],
) -> dict[str, dict]:
    if not selected_experiments:
        return experiment_inputs

    missing = [name for name in selected_experiments if name not in experiment_inputs]
    if missing:
        available = ", ".join(sorted(experiment_inputs))
        raise SystemExit(
            "Requested experiment(s) not found or invalid: "
            f"{', '.join(missing)}. Available: {available}"
        )

    return {name: experiment_inputs[name] for name in selected_experiments}


def _resolve_inputs(in_path: Path, selected_experiments: list[str]) -> tuple[dict[str, dict], str]:
    if not in_path.is_dir():
        raise SystemExit(
            f"INPUT_PATH must be an experiment root directory containing per-experiment folders: {in_path}"
        )

    subdirs = sorted([p for p in in_path.iterdir() if p.is_dir()])
    if not subdirs:
        raise SystemExit("No experiment folders found in INPUT_PATH.")

    experiment_inputs: dict[str, dict] = {}
    invalid_folders: list[str] = []

    for exp_dir in subdirs:
        exp_name = exp_dir.name
        force_disp_csv = exp_dir / f"{exp_name}_force_disp.csv"
        summary_json = exp_dir / f"{exp_name}_summary.json"

        if force_disp_csv.is_file() and summary_json.is_file():
            experiment_inputs[exp_name] = {
                "mode": "force_disp",
                "force_disp_csv": force_disp_csv,
                "summary_json": summary_json,
            }
        else:
            missing_parts = []
            if not force_disp_csv.is_file():
                missing_parts.append(force_disp_csv.name)
            if not summary_json.is_file():
                missing_parts.append(summary_json.name)
            invalid_folders.append(f"{exp_name} (missing: {', '.join(missing_parts)})")

    if invalid_folders:
        preview = ", ".join(invalid_folders[:5])
        raise SystemExit(
            "Found experiment folder(s) missing required force-disp inputs: "
            f"{preview}{' ...' if len(invalid_folders) > 5 else ''}"
        )

    if not experiment_inputs:
        raise SystemExit("No valid experiment folders found in INPUT_PATH.")

    return _filter_experiment_inputs(experiment_inputs, selected_experiments), in_path.name


def _read_force_disp_inputs(
    force_disp_csv: Path,
    summary_json: Path,
) -> tuple[pd.DataFrame, float, float]:
    force_disp = pd.read_csv(force_disp_csv)
    force_disp = force_disp.rename(columns={c: c.strip() for c in force_disp.columns})

    required = ["displacement_m", "force_N"]
    missing = [c for c in required if c not in force_disp.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {force_disp_csv}: {missing}. "
            f"Found: {list(force_disp.columns)}"
        )

    out = force_disp[required].copy().dropna()
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna()
    if out.empty:
        raise ValueError(f"No numeric rows found in {force_disp_csv} after parsing {required}.")

    summary = json.loads(summary_json.read_text())
    processing = summary.get("processing", {})
    l0_m = float(processing.get("L0_m", float("nan")))
    f_thresh_n = float(processing.get("F_thresh_N", float("nan")))
    if not np.isfinite(l0_m) or l0_m <= 0:
        raise ValueError(f"Invalid L0_m in {summary_json}. Cannot compute strain.")

    return out, l0_m, f_thresh_n


# Process all configured input files.
def main() -> None:
    args = _parse_args()
    experiment_inputs, batch_name = _resolve_inputs(INPUT_PATH, args.experiments)

    generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    code_version = "unknown"
    try:
        code_version = f"git:{subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()}"
    except Exception:
        pass
    for experiment_name, exp_input in experiment_inputs.items():
        force_disp, l0_m, f_thresh_n = _read_force_disp_inputs(
            exp_input["force_disp_csv"],
            exp_input["summary_json"],
        )
        final, summary, full = compute_engineering_from_force_disp(
            force_disp,
            l0_m=l0_m,
            width_mm=WIDTH_MM,
            thickness_mm=THICKNESS_MM,
            f_thresh_n=f_thresh_n,
        )
        preprocessing = pd.DataFrame(
            {
                "time_s": np.arange(len(force_disp), dtype=float),
                "marker_dist_m": force_disp["displacement_m"].to_numpy(float) + l0_m,
                "force_N": force_disp[FORCE_COLUMN].to_numpy(float),
            }
        )
        source_name = exp_input["force_disp_csv"].name

        summary_out = {
            "test_group": batch_name,
            "experiment": experiment_name,
            "generated_at_utc": generated_at_utc,
            "code_version": code_version,
            **summary,
        }
        save_outputs(OUT_ROOT, experiment_name, preprocessing, full, final, summary_out, export_debug=EXPORT_DEBUG)
        counts = summary["processing"]
        print(
            f"Processed: {experiment_name} ({source_name})  "
            f"-> wrote {counts['n_rows_output']}/{counts['n_rows_preprocessing']} rows"
        )


if __name__ == "__main__":
    main()
