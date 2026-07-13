"""Convert preprocessed tensile CSV data into engineering stress-strain results.

Edit the configuration constants below, then run this file directly. Each input
CSV is validated, L0 is estimated, engineering stress and strain are calculated,
and the processed CSV and JSON summary are written to disk.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration: edit these constants before running the script.
INPUT_PATH = Path("data/raw_input")  # One combined CSV or a folder of combined CSV files.
OUT_ROOT = Path("data/processed_stress_strain")
WIDTH_MM = 10.0
THICKNESS_MM = 2.0
F_THRESH_MIN_N = 1.0
F_THRESH_FRAC_OF_MAX = 0.01
EXPORT_DEBUG = False

# Required input column names.
TIME_COLUMN = "time_s"
DISTANCE_COLUMN = "marker_dist_m"
FORCE_COLUMN = "force_N"
REQUIRED_COLUMNS = [TIME_COLUMN, DISTANCE_COLUMN, FORCE_COLUMN]


# Read, validate, and clean one preprocessing CSV.
def read_preprocessing_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file type: {path.suffix}. Only .csv is supported.")

    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    out = df[REQUIRED_COLUMNS].copy().dropna()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()

    return out


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
        "toughness_MJ_per_m3": toughness,
        **modulus_info,
    }


# Calculate L0, engineering stress/strain, and summary metrics.
def compute_engineering(
    preprocessing: pd.DataFrame,
    width_mm: float,
    thickness_mm: float,
    f_thresh_min_N: float = 1.0,
    f_thresh_frac_of_max: float = 0.01,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    time_s = preprocessing[TIME_COLUMN].to_numpy(float)
    marker_dist_m = preprocessing[DISTANCE_COLUMN].to_numpy(float)
    force_N = preprocessing[FORCE_COLUMN].to_numpy(float)

    area_m2 = (width_mm / 1000.0) * (thickness_mm / 1000.0)

    f_max = float(np.nanmax(force_N)) if len(force_N) else float("nan")
    f_thresh = float(max(f_thresh_min_N, f_thresh_frac_of_max * f_max))

    low_mask = force_N <= f_thresh
    if low_mask.sum() >= 3:
        l0_m = float(np.nanmedian(marker_dist_m[low_mask]))
    else:
        l0_m = float(np.nanmedian(marker_dist_m[: min(10, len(marker_dist_m))]))

    strain_eng = (marker_dist_m - l0_m) / l0_m
    stress_MPa = (force_N / area_m2) / 1e6

    full = pd.DataFrame(
        {
            "time_s": time_s,
            "marker_dist_m": marker_dist_m,
            "force_N": force_N,
            "strain_eng": strain_eng,
            "stress_eng_MPa": stress_MPa,
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
            "F_thresh_N": f_thresh,
            "L0_m": l0_m,
            "n_rows_preprocessing": int(len(preprocessing)),
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
        preprocessing.to_csv(out_dir / f"{stem}_preprocessing_export.csv", index=False)
        full.to_csv(out_dir / f"{stem}_processed_full.csv", index=False)
    final.to_csv(out_dir / f"{stem}_final_export.csv", index=False)

    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))


# Resolve a single CSV or a directory of CSV files into a processing batch.
def _resolve_inputs(in_path: Path) -> tuple[list[Path], str]:
    if in_path.is_dir():
        files = sorted([p for p in in_path.iterdir() if p.suffix.lower() == ".csv"])
        if not files:
            raise SystemExit("No .csv files found in input folder.")
        return files, in_path.name

    return [in_path], in_path.stem


# Process all configured input files.
def main() -> None:
    files, batch_name = _resolve_inputs(INPUT_PATH)

    generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    code_version = "unknown"
    try:
        code_version = f"git:{subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()}"
    except Exception:
        pass
    for f in files:
        preprocessing = read_preprocessing_table(f)
        final, summary, full = compute_engineering(
            preprocessing,
            width_mm=WIDTH_MM,
            thickness_mm=THICKNESS_MM,
            f_thresh_min_N=F_THRESH_MIN_N,
            f_thresh_frac_of_max=F_THRESH_FRAC_OF_MAX,
        )
        summary_out = {
            "test_group": batch_name,
            "generated_at_utc": generated_at_utc,
            "code_version": code_version,
            **summary,
        }
        save_outputs(OUT_ROOT, f.stem, preprocessing, full, final, summary_out, export_debug=EXPORT_DEBUG)
        counts = summary["processing"]
        print(f"Processed: {f.name}  -> wrote {counts['n_rows_output']}/{counts['n_rows_preprocessing']} rows")


if __name__ == "__main__":
    main()
