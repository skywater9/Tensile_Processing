"""Convert preprocessed tensile CSV data into force-displacement results.

Edit the configuration constants below, then run this file directly. Each input
CSV is validated, L0 is estimated from the low-force region, displacement is
calculated, and the processed CSV and JSON summary are written to disk.
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
OUT_ROOT = Path("data/processed_force_disp")
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


# Calculate L0, displacement, and force-displacement summary metrics.
def compute_force_displacement(
    preprocessing: pd.DataFrame,
    f_thresh_min_N: float = 1.0,
    f_thresh_frac_of_max: float = 0.01,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    time_s = preprocessing[TIME_COLUMN].to_numpy(float)
    marker_dist_m = preprocessing[DISTANCE_COLUMN].to_numpy(float)
    force_N = preprocessing[FORCE_COLUMN].to_numpy(float)

    f_max = float(np.nanmax(force_N)) if len(force_N) else float("nan")
    f_thresh = float(max(f_thresh_min_N, f_thresh_frac_of_max * f_max))

    low_mask = force_N <= f_thresh
    if low_mask.sum() >= 3:
        l0_m = float(np.nanmedian(marker_dist_m[low_mask]))
    else:
        l0_m = float(np.nanmedian(marker_dist_m[: min(10, len(marker_dist_m))]))

    displacement_m = marker_dist_m - l0_m

    full = pd.DataFrame(
        {
            "time_s": time_s,
            "marker_dist_m": marker_dist_m,
            "force_N": force_N,
            "displacement_m": displacement_m,
        }
    )

    final = full[["displacement_m", "force_N"]].copy().dropna()

    metrics = {
        "max_force_N": float(np.nanmax(force_N)) if len(force_N) else float("nan"),
        "max_displacement_m": float(np.nanmax(displacement_m)) if len(displacement_m) else float("nan"),
    }

    summary = {
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
    final.to_csv(out_dir / f"{stem}_force_disp.csv", index=False)

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
        final, summary, full = compute_force_displacement(
            preprocessing,
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
