from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["time_s", "marker_dist_m", "force_N"]


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


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = m*x + b and return (m, b, r2)."""
    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return float(m), float(b), float(r2)


def _find_apparent_modulus_window(
    strain: np.ndarray,
    stress: np.ndarray,
    *,
    min_points: int = 8,
    min_strain_span: float = 8e-4,
    stress_low_frac: float = 0.10,
    stress_high_frac: float = 0.45,
) -> dict:
    """
    Find a data-driven apparent modulus window in the early stress range.
    Returns apparent_Youngs_modulus_GPa and apparent_Youngs_modulus_window_MPa.
    """
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


def _compute_metrics(full: pd.DataFrame) -> dict:
    """Compute summary metrics from kept region only."""
    kept = full.loc[full["keep_flag"]].copy()
    kept = kept[np.isfinite(kept["strain_eng"]) & np.isfinite(kept["stress_eng_MPa"])]

    if len(kept) < 5:
        return {
            "UTS_MPa": float("nan"),
            "strain_at_UTS": float("nan"),
            "max_strain_eng": float("nan"),
            "toughness_MJ_per_m3": float("nan"),
            "apparent_Youngs_modulus_GPa": float("nan"),
            "apparent_Youngs_modulus_window_MPa": [float("nan"), float("nan")],
        }

    kept = kept.sort_values("strain_eng")
    strain = kept["strain_eng"].to_numpy(float)
    stress = kept["stress_eng_MPa"].to_numpy(float)

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


def compute_engineering(
    raw: pd.DataFrame,
    width_mm: float,
    thickness_mm: float,
    f_thresh_min_N: float = 1.0,
    f_thresh_frac_of_max: float = 0.01,
    keep_multiplier: float = 2.0,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Compute strain/stress, keep_flag, and summary metrics."""
    time_s = raw["time_s"].to_numpy(float)
    marker_dist_m = raw["marker_dist_m"].to_numpy(float)
    force_N = raw["force_N"].to_numpy(float)

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

    keep_flag = force_N >= (keep_multiplier * f_thresh)

    full = pd.DataFrame(
        {
            "time_s": time_s,
            "marker_dist_m": marker_dist_m,
            "force_N": force_N,
            "strain_eng": strain_eng,
            "stress_eng_MPa": stress_MPa,
            "keep_flag": keep_flag,
        }
    )

    final = full.loc[keep_flag, ["strain_eng", "stress_eng_MPa"]].copy()
    final = final.dropna().sort_values("strain_eng")
    final = final.groupby("strain_eng", as_index=False)["stress_eng_MPa"].mean()

    metrics = _compute_metrics(full)

    summary = {
        "specimen_geometry": {
            "width_mm": width_mm,
            "thickness_mm": thickness_mm,
            "area_m2": area_m2,
        },
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
    final.to_csv(out_dir / f"{stem}_final_export.csv", index=False)

    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))

    plt.figure()
    plt.plot(full["strain_eng"], full["stress_eng_MPa"], alpha=0.25, label="all points")
    plt.plot(final["strain_eng"], final["stress_eng_MPa"], linewidth=2.0, label="trimmed (final)")
    plt.xlabel("Engineering strain")
    plt.ylabel("Engineering stress (MPa)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_stress_strain.png", dpi=200)
    plt.close()


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
    ap = argparse.ArgumentParser(description="Process raw tensile data into trimmed engineering stress-strain.")
    ap.add_argument("--input", required=True, help="Path to raw .csv file OR a folder of .csv files.")
    ap.add_argument("--out", default="data\\processed", help="Output directory.")
    ap.add_argument("--width-mm", type=float, required=True, help="Specimen width in mm.")
    ap.add_argument("--thickness-mm", type=float, required=True, help="Specimen thickness in mm.")
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
    out_dir = Path(args.out)

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
        final, summary, full = compute_engineering(
            raw,
            width_mm=args.width_mm,
            thickness_mm=args.thickness_mm,
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
