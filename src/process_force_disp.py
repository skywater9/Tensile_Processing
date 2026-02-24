from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["time_s", "marker_dist_m", "force_N"]
OUTPUT_SUBDIR = "force_disp"


def read_raw_table(path: Path) -> pd.DataFrame:
    """Read raw data from CSV. Returns df with columns: time_s, marker_dist_m, force_N."""
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


def compute_force_displacement(
    raw: pd.DataFrame,
    f_thresh_min_N: float = 1.0,
    f_thresh_frac_of_max: float = 0.01,
    keep_multiplier: float = 2.0,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Compute displacement, keep_flag, and summary metrics."""
    time_s = raw["time_s"].to_numpy(float)
    marker_dist_m = raw["marker_dist_m"].to_numpy(float)
    force_N = raw["force_N"].to_numpy(float)

    f_max = float(np.nanmax(force_N)) if len(force_N) else float("nan")
    f_thresh = float(max(f_thresh_min_N, f_thresh_frac_of_max * f_max))

    low_mask = force_N <= f_thresh
    if low_mask.sum() >= 3:
        l0_m = float(np.nanmedian(marker_dist_m[low_mask]))
    else:
        l0_m = float(np.nanmedian(marker_dist_m[: min(10, len(marker_dist_m))]))

    displacement_m = marker_dist_m - l0_m

    keep_flag = force_N >= (keep_multiplier * f_thresh)

    full = pd.DataFrame(
        {
            "time_s": time_s,
            "marker_dist_m": marker_dist_m,
            "force_N": force_N,
            "displacement_m": displacement_m,
            "keep_flag": keep_flag,
        }
    )

    final = full.loc[keep_flag, ["displacement_m", "force_N"]].copy()
    final = final.dropna().sort_values("displacement_m")
    final = final.groupby("displacement_m", as_index=False)["force_N"].mean()

    metrics = {
        "max_force_N": float(np.nanmax(force_N)) if len(force_N) else float("nan"),
        "max_displacement_m": float(np.nanmax(displacement_m)) if len(displacement_m) else float("nan"),
    }

    summary = {
        "processing": {
            "F_thresh_N": f_thresh,
            "keep_multiplier": keep_multiplier,
            "L0_m": l0_m,
            "n_rows_raw": int(len(raw)),
            "n_rows_kept": int(keep_flag.sum()),
        },
        "metrics": metrics,
    }

    return final, summary, full


def save_outputs(
    out_root: Path,
    stem: str,
    raw: pd.DataFrame,
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
        raw.to_csv(out_dir / f"{stem}_raw_export.csv", index=False)
        full.to_csv(out_dir / f"{stem}_processed_full.csv", index=False)
    final.to_csv(out_dir / f"{stem}_force_disp.csv", index=False)

    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))

def _resolve_inputs(in_path: Path) -> tuple[list[Path], str]:
    if in_path.is_dir():
        files = sorted([p for p in in_path.iterdir() if p.suffix.lower() == ".csv"])
        if not files:
            raise SystemExit("No .csv files found in input folder.")
        return files, in_path.name

    return [in_path], in_path.stem


def _prepare_batch_dir(out_dir: Path, batch_name: str) -> Path:
    batch_dir = out_dir / f"{batch_name}"
    if batch_dir.exists():
        if batch_dir.is_dir():
            shutil.rmtree(batch_dir)
        else:
            batch_dir.unlink()
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Process raw tensile data into force-displacement outputs.")
    ap.add_argument("--input", required=True, help="Path to raw .csv file OR a folder of .csv files.")
    ap.add_argument(
        "--out",
        default="data\\processed",
        help=f"Output root directory. Results are written under <out>\\{OUTPUT_SUBDIR}.",
    )
    ap.add_argument("--fth-min", type=float, default=1.0, help="Minimum force threshold in N for L0 detection.")
    ap.add_argument("--fth-frac", type=float, default=0.01, help="Force threshold as fraction of max force.")
    ap.add_argument("--keep-mult", type=float, default=2.0, help="keep_flag condition: F >= keep_mult * F_thresh.")
    ap.add_argument(
        "--export-debug",
        action="store_true",
        help="Write raw_export and processed_full CSVs for debugging.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out) / OUTPUT_SUBDIR

    files, batch_name = _resolve_inputs(in_path)
    batch_dir = _prepare_batch_dir(out_dir, batch_name)

    generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    code_version = "unknown"
    try:
        import subprocess

        code_version = f"git:{subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()}"
    except Exception:
        pass
    for f in files:
        raw = read_raw_table(f)
        final, summary, full = compute_force_displacement(
            raw,
            f_thresh_min_N=args.fth_min,
            f_thresh_frac_of_max=args.fth_frac,
            keep_multiplier=args.keep_mult,
        )
        summary_out = {
            "test_group": batch_name,
            "generated_at_utc": generated_at_utc,
            "code_version": code_version,
            **summary,
        }
        save_outputs(batch_dir, f.stem, raw, full, final, summary_out, export_debug=args.export_debug)
        counts = summary["processing"]
        print(f"Processed: {f.name}  -> kept {counts['n_rows_kept']}/{counts['n_rows_raw']} rows")


if __name__ == "__main__":
    main()
