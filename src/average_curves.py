from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_final_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "strain_eng" not in df.columns or "stress_eng_MPa" not in df.columns:
        raise ValueError(f"{path.name} must have columns: strain_eng, stress_eng_MPa")
    df = df[["strain_eng", "stress_eng_MPa"]].dropna().sort_values("strain_eng")
    df = df.groupby("strain_eng", as_index=False)["stress_eng_MPa"].mean()
    return df


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
    """Find a data-driven apparent modulus window in the early stress range."""
    finite = np.isfinite(strain) & np.isfinite(stress)
    strain = strain[finite]
    stress = stress[finite]

    if len(strain) < min_points:
        return {
            "apparent_Youngs_modulus_GPa": float("nan"),
            "apparent_Youngs_modulus_window_MPa": [float("nan"), float("nan")],
            "apparent_Youngs_modulus_window_low_MPa": float("nan"),
            "apparent_Youngs_modulus_window_high_MPa": float("nan"),
        }

    uts = float(np.nanmax(stress))
    if not np.isfinite(uts) or uts <= 0:
        return {
            "apparent_Youngs_modulus_GPa": float("nan"),
            "apparent_Youngs_modulus_window_MPa": [float("nan"), float("nan")],
            "apparent_Youngs_modulus_window_low_MPa": float("nan"),
            "apparent_Youngs_modulus_window_high_MPa": float("nan"),
        }

    s_lo = stress_low_frac * uts
    s_hi = stress_high_frac * uts
    idx = np.flatnonzero((stress >= s_lo) & (stress <= s_hi))
    if len(idx) < min_points:
        return {
            "apparent_Youngs_modulus_GPa": float("nan"),
            "apparent_Youngs_modulus_window_MPa": [s_lo, s_hi],
            "apparent_Youngs_modulus_window_low_MPa": float(s_lo),
            "apparent_Youngs_modulus_window_high_MPa": float(s_hi),
        }

    strain_c = strain[idx]
    stress_c = stress[idx]

    best = None  # (score, m, smin, smax)
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
                best = (score, m, float(y.min()), float(y.max()))

    if best is None:
        return {
            "apparent_Youngs_modulus_GPa": float("nan"),
            "apparent_Youngs_modulus_window_MPa": [s_lo, s_hi],
            "apparent_Youngs_modulus_window_low_MPa": float(s_lo),
            "apparent_Youngs_modulus_window_high_MPa": float(s_hi),
        }

    _, m_best, s_min, s_max = best
    return {
        "apparent_Youngs_modulus_GPa": float(m_best / 1000.0),
        "apparent_Youngs_modulus_window_MPa": [s_min, s_max],
        "apparent_Youngs_modulus_window_low_MPa": float(s_min),
        "apparent_Youngs_modulus_window_high_MPa": float(s_max),
    }


def _curve_metrics(curve: pd.DataFrame) -> dict:
    """Compute per-curve metrics that mirror raw processing summary fields."""
    strain = curve["strain_eng"].to_numpy(float)
    stress = curve["stress_eng_MPa"].to_numpy(float)

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Average multiple trimmed stress-strain curves.")
    ap.add_argument("--input", required=True, help="Batch output folder (e.g., data\\processed\\1_A).")
    ap.add_argument("--out", default="data\\results", help="Base output directory.")
    ap.add_argument("--n-grid", type=int, default=3000, help="Number of strain points for averaging grid.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_dir():
        raise SystemExit("Input must be a batch folder containing per-sample subfolders.")

    out_dir = Path(args.out) / in_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for subdir in sorted([p for p in in_path.iterdir() if p.is_dir()]):
        candidate = subdir / f"{subdir.name}_final_export.csv"
        if candidate.exists():
            files.append(candidate)

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
    avg.to_csv(out_dir / "avg_curve.csv", index=False)

    per_rows = []
    for f, c in zip(files, curves):
        per_rows.append({"file": f.name, **_curve_metrics(c)})

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(out_dir / "per_test_summary.csv", index=False)

    summary = {
        "UTS_MPa": float(per_df["UTS_MPa"].mean()),
        "UTS_MPa_std": float(per_df["UTS_MPa"].std(ddof=1)),
        "strain_at_UTS": float(per_df["strain_at_UTS"].mean()),
        "strain_at_UTS_std": float(per_df["strain_at_UTS"].std(ddof=1)),
        "max_strain_eng": float(per_df["max_strain_eng"].mean()),
        "max_strain_eng_std": float(per_df["max_strain_eng"].std(ddof=1)),
        "toughness_MJ_per_m3": float(per_df["toughness_MJ_per_m3"].mean()),
        "toughness_MJ_per_m3_std": float(per_df["toughness_MJ_per_m3"].std(ddof=1)),
        "apparent_Youngs_modulus_GPa": float(per_df["apparent_Youngs_modulus_GPa"].mean()),
        "apparent_Youngs_modulus_GPa_std": float(per_df["apparent_Youngs_modulus_GPa"].std(ddof=1)),
        "apparent_Youngs_modulus_window_MPa": [
            float(per_df["apparent_Youngs_modulus_window_MPa"].apply(lambda v: v[0]).mean()),
            float(per_df["apparent_Youngs_modulus_window_MPa"].apply(lambda v: v[1]).mean()),
        ],
        "apparent_Youngs_modulus_window_MPa_std": [
            float(per_df["apparent_Youngs_modulus_window_MPa"].apply(lambda v: v[0]).std(ddof=1)),
            float(per_df["apparent_Youngs_modulus_window_MPa"].apply(lambda v: v[1]).std(ddof=1)),
        ],
        "n_tests": len(files),
        "strain_min_common": float(eps_min_common),
        "strain_max_common": float(eps_max_common),
        "summary_source": "curve_inferred_unweighted",
    }
    (out_dir / "avg_summary.json").write_text(json.dumps(summary, indent=2))

    # Plot: individual + mean + +/-1 sigma band
    plt.figure()
    for c in curves:
        plt.plot(c["strain_eng"], c["stress_eng_MPa"], alpha=0.35)
    plt.plot(grid, mean, linewidth=2.5, label="mean")
    plt.fill_between(grid, mean - std, mean + std, alpha=0.2, label="+/-1 sigma")
    plt.xlabel("Engineering strain")
    plt.ylabel("Engineering stress (MPa)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "avg_plot.png", dpi=250)
    plt.close()

    print(f"Wrote: {out_dir/'avg_curve.csv'}")
    print(f"Wrote: {out_dir/'per_test_summary.csv'}")
    print(f"Wrote: {out_dir/'avg_summary.json'}")
    print(f"Wrote: {out_dir/'avg_plot.png'}")


if __name__ == "__main__":
    main()
