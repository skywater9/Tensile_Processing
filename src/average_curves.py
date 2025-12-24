from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_final_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "strain_eng" not in df.columns or "stress_eng_MPa" not in df.columns:
        raise ValueError(f"{path.name} must have columns: strain_eng, stress_eng_MPa")
    df = df[["strain_eng", "stress_eng_MPa"]].dropna().sort_values("strain_eng")
    df = df.groupby("strain_eng", as_index=False)["stress_eng_MPa"].mean()
    return df


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def _get_code_version() -> str:
    try:
        import subprocess

        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return f"git:{sha}"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Average multiple trimmed stress-strain curves.")
    ap.add_argument("--input", required=True, help="Batch output folder (e.g., data\\processed\\1_A).")
    ap.add_argument("--out", default="results", help="Base output directory.")
    ap.add_argument("--n-grid", type=int, default=3000, help="Number of strain points for averaging grid.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_dir():
        raise SystemExit("Input must be a batch folder containing per-sample subfolders.")

    batch_name = in_path.name
    out_dir = Path(args.out) / batch_name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    summaries = []
    for subdir in sorted([p for p in in_path.iterdir() if p.is_dir()]):
        stem = subdir.name
        curve_path = subdir / f"{stem}_final_export.csv"
        summary_path = subdir / f"{stem}_summary.json"
        if curve_path.exists() and summary_path.exists():
            files.append(curve_path)
            summaries.append(load_summary(summary_path))

    if len(files) < 2:
        raise SystemExit("Need at least 2 final_export curves in batch subfolders to average.")

    curves = [load_final_curve(f) for f in files]

    eps_max_common = min(c["strain_eng"].max() for c in curves)
    eps_min_common = max(0.0, max(c["strain_eng"].min() for c in curves))

    grid = np.linspace(float(eps_min_common), float(eps_max_common), args.n_grid)

    interp = np.vstack(
        [np.interp(grid, c["strain_eng"].to_numpy(), c["stress_eng_MPa"].to_numpy()) for c in curves]
    )

    mean = interp.mean(axis=0)
    std = interp.std(axis=0, ddof=1)

    avg = pd.DataFrame(
        {
            "strain_eng": grid,
            "stress_mean_MPa": mean,
            "stress_std_MPa": std,
            "stress_plus_1sigma_MPa": mean + std,
            "stress_minus_1sigma_MPa": mean - std,
        }
    )
    avg.to_csv(out_dir / f"{batch_name}_curve.csv", index=False)

    per_rows = []
    for f, summary in zip(files, summaries):
        row = {"file": f.name}
        row.update(summary.get("metrics", {}))
        per_rows.append(row)

    per_df = pd.DataFrame(per_rows)

    metrics_block = summaries[0].get("metrics", {})
    metrics_out = {}
    for key in metrics_block.keys():
        if key == "apparent_Youngs_modulus_window_MPa":
            continue
        if key not in per_df.columns:
            continue
        series = per_df[key]
        sample = series.dropna().iloc[0] if not series.dropna().empty else None
        if isinstance(sample, list) and len(sample) == 2:
            vals = np.vstack(series.dropna().apply(lambda v: np.array(v, dtype=float)))
            means = vals.mean(axis=0)
            stds = vals.std(axis=0, ddof=1) if len(vals) > 1 else np.array([float("nan"), float("nan")])
            metrics_out[key] = {
                "mean": [float(means[0]), float(means[1])],
                "std": [float(stds[0]), float(stds[1])],
            }
        else:
            metrics_out[key] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
            }

    specimen_geometry = summaries[0].get("specimen_geometry", {})

    summary = {
        "test_group": batch_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "code_version": _get_code_version(),
        "batch_info": {
            "n_specimens": len(files),
            "strain_min_common": float(eps_min_common),
            "strain_max_common": float(eps_max_common),
            "averaging_method": "curve_resampled_unweighted",
        },
        "specimen_geometry": specimen_geometry,
        "metrics": metrics_out,
    }
    (out_dir / f"{batch_name}_summary.json").write_text(json.dumps(summary, indent=2))

    # Plot: mean + +/-1 sigma band
    plt.figure()
    plt.plot(grid, mean, linewidth=2.5, color="tab:blue", label=f"{batch_name} mean")
    plt.fill_between(
        grid,
        mean - std,
        mean + std,
        color="tab:blue",
        alpha=0.2,
        label=f"{batch_name} ±1 standard deviation (n={len(files)})",
    )
    plt.xlabel("Strain (-)")
    plt.ylabel("Stress (MPa)")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{batch_name}_plot.png", dpi=250)
    plt.close()

    print(f"Wrote: {out_dir/f'{batch_name}_curve.csv'}")
    print(f"Wrote: {out_dir/f'{batch_name}_summary.json'}")
    print(f"Wrote: {out_dir/f'{batch_name}_plot.png'}")


if __name__ == "__main__":
    main()
