#!/usr/bin/env python3
"""
PLE (Progressive Layered Extraction MoE) vs Plain (dense) -- training log analysis.

Expected files (same folder as this script, or set DATA_DIR):
    thesis_cache_{plain,ple}_{2,4,8,16,32}D.csv

Install:  pip install pandas numpy matplotlib scipy
Run:      python analyze.py

Outputs go to ./analysis_output/ (PNG figures + CSV tables).

Notes
-----
* Final metrics are taken at the checkpoint with the lowest eval loss
  (not the last step), so the test set is not used for model selection.
* OVERALL is an average over a different target set at each D, so it is NOT
  comparable across D. Cross-D comparisons use only targets present in
  every D (e.g. D9, E2).
* One run per setting means no seed variance. Treat small gaps with caution.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    from scipy.stats import wilcoxon
except ImportError:  # scipy optional
    wilcoxon = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_DIR = Path(".")
OUT_DIR = Path("analysis_output")
DIMS = [2, 4, 8, 16, 32]
MODELS = ["plain", "ple"]
LABEL = {"plain": "Dense (plain)", "ple": "PLE (MoE)"}
COLOR = {"plain": "#4C72B0", "ple": "#DD8452"}
SMOOTH = 15          # rolling window (logged points) for noisy train curves
LAST_N = 5           # last N eval points used for mean +- std
CONV_FRAC = 0.05     # "converged" = eval loss within 5% of its minimum
LOG_LOSS = True      # log-scale y axis for loss curves

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


# ----------------------------------------------------------------------------
# LOADING + HELPERS
# ----------------------------------------------------------------------------
def load_all():
    data = {}
    for m in MODELS:
        for d in DIMS:
            p = DATA_DIR / f"thesis_cache_{m}_{d}D.csv"
            if p.exists():
                data[(m, d)] = pd.read_csv(p).sort_values("step").reset_index(drop=True)
            else:
                print(f"[warn] missing file: {p}")
    return data


def series(df, col, smooth=0):
    """Non-NaN series indexed by step, optionally smoothed."""
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = df.set_index("step")[col].dropna().sort_index()
    if smooth and smooth > 1:
        s = s.rolling(smooth, min_periods=1, center=True).mean()
    return s


def get_targets(df):
    """Target names (without OVERALL) inferred from eval/r2_* columns."""
    out = []
    for c in df.columns:
        m = re.match(r"^eval/r2_(.+)$", c)
        if m and m.group(1) != "OVERALL":
            out.append(m.group(1))
    return out


def final_metrics(df):
    """Metrics at the best-eval-loss checkpoint (eval/test forward-filled)."""
    ev = df.dropna(subset=["eval/loss"])
    if ev.empty:
        return None
    best_step = ev.loc[ev["eval/loss"].idxmin(), "step"]
    ff = df.copy()
    cols = [c for c in ff.columns if c.startswith(("eval/", "test/"))]
    ff[cols] = ff[cols].ffill()
    row = ff[ff["step"] == best_step].iloc[0]
    label = "best-eval"
    if pd.isna(row.get("test/r2_OVERALL", np.nan)):
        row = ff.iloc[-1]
        label = "last-step (test not logged at best-eval step)"
    return {"step": int(best_step), "row": row, "label": label}


def steps_to_within(df, frac=CONV_FRAC):
    ev = series(df, "eval/loss", smooth=3)
    if ev.empty:
        return np.nan
    thr = ev.min() * (1 + frac)
    return int(ev[ev <= thr].index[0])


def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png")


# ----------------------------------------------------------------------------
# TABLES
# ----------------------------------------------------------------------------
def overall_table(data):
    rows = []
    for (m, d), df in sorted(data.items(), key=lambda x: (x[0][1], x[0][0])):
        fm = final_metrics(df)
        if fm is None:
            continue
        r = fm["row"]
        tr = series(df, "train/loss", smooth=SMOOTH)
        te_r2 = series(df, "test/r2_OVERALL").tail(LAST_N)
        rows.append({
            "dim": d, "model": m, "n_targets": len(get_targets(df)),
            "checkpoint": fm["label"], "best_step": fm["step"],
            "total_steps": int(df["step"].max()),
            "steps_to_within_5pct": steps_to_within(df),
            "train_loss_end": tr.iloc[-1] if len(tr) else np.nan,
            "eval_loss_best": r.get("eval/loss", np.nan),
            "eval_mse": r.get("eval/true_avg_mse_OVERALL", np.nan),
            "eval_r2": r.get("eval/r2_OVERALL", np.nan),
            "test_mse": r.get("test/true_avg_mse_OVERALL", np.nan),
            "test_r2": r.get("test/r2_OVERALL", np.nan),
            "test_r2_lastN_mean": te_r2.mean() if len(te_r2) else np.nan,
            "test_r2_lastN_std": te_r2.std() if len(te_r2) > 1 else np.nan,
            "gap_eval_minus_train": (r.get("eval/loss", np.nan) - tr.iloc[-1]) if len(tr) else np.nan,
        })
    return pd.DataFrame(rows)


def per_target_table(data):
    rows = []
    for (m, d), df in data.items():
        fm = final_metrics(df)
        if fm is None:
            continue
        r = fm["row"]
        for t in get_targets(df):
            rows.append({
                "model": m, "dim": d, "target": t,
                "test_r2": r.get(f"test/r2_{t}", np.nan),
                "test_mse": r.get(f"test/true_avg_mse_{t}", np.nan),
                "eval_r2": r.get(f"eval/r2_{t}", np.nan),
                "eval_mse": r.get(f"eval/true_avg_mse_{t}", np.nan),
            })
    return pd.DataFrame(rows)


def stats_table(pt):
    rows = []
    for d, g in pt.groupby("dim"):
        w = g.pivot(index="target", columns="model", values="test_r2").dropna()
        if not {"ple", "plain"}.issubset(w.columns) or w.empty:
            continue
        delta = w["ple"] - w["plain"]
        p = np.nan
        if wilcoxon is not None and len(delta) >= 6 and (delta != 0).any():
            try:
                p = wilcoxon(delta).pvalue
            except ValueError:
                pass
        rows.append({
            "dim": d, "n_targets": len(delta),
            "mean_delta_r2": delta.mean(), "median_delta_r2": delta.median(),
            "ple_wins": int((delta > 0).sum()), "plain_wins": int((delta < 0).sum()),
            "wilcoxon_p": p,
        })
    return pd.DataFrame(rows).sort_values("dim")


# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------
def plot_loss_curves(data):
    rows = [("train/loss", "Train loss", SMOOTH),
            ("eval/loss", "Eval loss", 1),
            ("test/loss", "Test loss", 1)]
    fig, axes = plt.subplots(len(rows), len(DIMS),
                             figsize=(3.6 * len(DIMS), 3 * len(rows)), squeeze=False)
    for j, d in enumerate(DIMS):
        for i, (col, name, sm) in enumerate(rows):
            ax = axes[i][j]
            for m in MODELS:
                df = data.get((m, d))
                if df is None:
                    continue
                s = series(df, col, sm)
                if not s.empty:
                    ax.plot(s.index, s.values, color=COLOR[m], lw=1.4, label=LABEL[m])
            if LOG_LOSS:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"{d}D targets")
            if j == 0:
                ax.set_ylabel(name)
            if i == len(rows) - 1:
                ax.set_xlabel("step")
    axes[0][0].legend(frameon=False)
    fig.suptitle("Loss curves: PLE vs dense", y=1.01, fontsize=13)
    save(fig, "01_loss_curves")


def plot_r2_curves(data):
    fig, axes = plt.subplots(1, len(DIMS), figsize=(3.6 * len(DIMS), 3.2), squeeze=False)
    for j, d in enumerate(DIMS):
        ax = axes[0][j]
        for m in MODELS:
            df = data.get((m, d))
            if df is None:
                continue
            s = series(df, "eval/r2_OVERALL")
            if not s.empty:
                ax.plot(s.index, s.values, color=COLOR[m], lw=1.4, label=LABEL[m])
        ax.set_title(f"{d}D")
        ax.set_xlabel("step")
        if j == 0:
            ax.set_ylabel("eval R2 (overall)")
    axes[0][0].legend(frameon=False)
    save(fig, "02_eval_r2_curves")


def plot_gap(data):
    fig, axes = plt.subplots(1, len(DIMS), figsize=(3.6 * len(DIMS), 3.2), squeeze=False)
    for j, d in enumerate(DIMS):
        ax = axes[0][j]
        for m in MODELS:
            df = data.get((m, d))
            if df is None:
                continue
            tr = series(df, "train/loss", smooth=SMOOTH)
            ev = series(df, "eval/loss")
            if tr.empty or ev.empty:
                continue
            gap = ev.values - np.interp(ev.index, tr.index, tr.values)
            ax.plot(ev.index, gap, color=COLOR[m], lw=1.4, label=LABEL[m])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{d}D")
        ax.set_xlabel("step")
        if j == 0:
            ax.set_ylabel("eval loss - train loss")
    axes[0][0].legend(frameon=False)
    fig.suptitle("Generalization gap over training", y=1.03, fontsize=13)
    save(fig, "03_generalization_gap")


def plot_overall_bars(ov):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(DIMS))
    w = 0.38
    for ax, col, title in [(axes[0], "test_mse", "Test MSE (overall) -- lower is better"),
                           (axes[1], "test_r2", "Test R2 (overall) -- higher is better")]:
        for k, m in enumerate(MODELS):
            vals = [ov[(ov.dim == d) & (ov.model == m)][col].squeeze()
                    if len(ov[(ov.dim == d) & (ov.model == m)]) else np.nan for d in DIMS]
            bars = ax.bar(x + (k - 0.5) * w, vals, w, color=COLOR[m], label=LABEL[m])
            for b, v in zip(bars, vals):
                if pd.notna(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}D" for d in DIMS])
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.text(0.5, -0.02, "OVERALL averages different target sets per D -- not comparable across D",
             ha="center", fontsize=8, style="italic")
    save(fig, "04_final_overall_bars")


def plot_scaling(pt):
    shared = None
    for d in DIMS:
        ts = set(pt[(pt.dim == d) & (pt.model == "plain")].target)
        if ts:
            shared = ts if shared is None else shared & ts
    shared = sorted(shared) if shared else []
    if not shared:
        print("  [skip] no shared targets across D")
        return
    panels = shared + ["MEAN (shared)"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.8), squeeze=False)
    for ax, t in zip(axes[0], panels):
        for m in MODELS:
            ys = []
            for d in DIMS:
                g = pt[(pt.dim == d) & (pt.model == m)]
                if t == "MEAN (shared)":
                    ys.append(g[g.target.isin(shared)].test_r2.mean())
                else:
                    v = g[g.target == t].test_r2
                    ys.append(v.iloc[0] if len(v) else np.nan)
            ax.plot(DIMS, ys, "o-", color=COLOR[m], label=LABEL[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks(DIMS)
        ax.set_xticklabels([f"{d}D" for d in DIMS])
        ax.set_title(t)
        ax.set_xlabel("number of target dimensions")
    axes[0][0].set_ylabel("test R2")
    axes[0][0].legend(frameon=False)
    fig.suptitle("Scaling with task count on shared targets (positive/negative transfer)",
                 y=1.03, fontsize=13)
    save(fig, "05_scaling_shared_targets")
    out = []
    for d in DIMS:
        for m in MODELS:
            g = pt[(pt.dim == d) & (pt.model == m) & (pt.target.isin(shared))]
            for _, r in g.iterrows():
                out.append({"dim": d, "model": m, "target": r.target, "test_r2": r.test_r2})
    pd.DataFrame(out).to_csv(OUT_DIR / "scaling_shared_targets.csv", index=False)


def plot_delta_heatmap(pt):
    w = pt.pivot_table(index=["dim", "target"], columns="model", values="test_r2").reset_index()
    if not {"ple", "plain"}.issubset(w.columns):
        return
    w["delta"] = w["ple"] - w["plain"]
    hm = w.pivot(index="target", columns="dim", values="delta")
    sort_col = max(hm.columns)
    hm = hm.sort_values(sort_col, ascending=False)
    a = np.nanmax(np.abs(hm.values))
    a = a if a > 0 else 1e-6
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e6e6e6")
    fig, ax = plt.subplots(figsize=(1.4 * len(hm.columns) + 3, 0.34 * len(hm) + 2))
    im = ax.imshow(hm.values, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -a, a))
    ax.set_xticks(range(len(hm.columns)))
    ax.set_xticklabels([f"{d}D" for d in hm.columns])
    ax.set_yticks(range(len(hm)))
    ax.set_yticklabels(hm.index)
    ax.grid(False)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="test R2: PLE - plain")
    ax.set_title("Per-target PLE advantage (blue = PLE better, red = plain better)")
    save(fig, "06_delta_r2_heatmap")


def plot_largest_dim(pt):
    d = max(pt.dim.unique())
    g = pt[pt.dim == d].pivot(index="target", columns="model", values="test_r2").dropna()
    if not {"ple", "plain"}.issubset(g.columns) or g.empty:
        return
    g["delta"] = g["ple"] - g["plain"]
    g = g.sort_values("delta", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, 0.28 * len(g) + 1.5)))

    ax = axes[0]
    ax.scatter(g["plain"], g["ple"], color=COLOR["ple"], s=40, zorder=3)
    lo = min(g["plain"].min(), g["ple"].min())
    hi = max(g["plain"].max(), g["ple"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    for t, r in g.iterrows():
        ax.annotate(t, (r["plain"], r["ple"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("plain test R2")
    ax.set_ylabel("PLE test R2")
    ax.set_title(f"{d}D: per-target R2 (above line = PLE better)")

    ax = axes[1]
    colors = [COLOR["ple"] if v > 0 else COLOR["plain"] for v in g["delta"]]
    ax.barh(g.index, g["delta"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("test R2: PLE - plain")
    ax.set_title(f"{d}D: ranked PLE improvement")

    ax = axes[2]
    mat = g[["plain", "ple"]].sort_values("plain", ascending=False)
    im = ax.imshow(mat.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LABEL["plain"], LABEL["ple"]])
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(2):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="test R2")
    ax.set_title(f"{d}D: R2 per target (hard -> easy)")
    save(fig, "07_largest_dim_per_target")


def plot_optimization(data):
    fig, axes = plt.subplots(2, len(DIMS), figsize=(3.6 * len(DIMS), 6), squeeze=False)
    for j, d in enumerate(DIMS):
        for m in MODELS:
            df = data.get((m, d))
            if df is None:
                continue
            g = series(df, "train/grad_norm", smooth=SMOOTH)
            lr = series(df, "train/lr")
            if not g.empty:
                axes[0][j].plot(g.index, g.values, color=COLOR[m], lw=1.2, label=LABEL[m])
            if not lr.empty:
                axes[1][j].plot(lr.index, lr.values, color=COLOR[m], lw=1.2, ls="--" if m == "ple" else "-")
        axes[0][j].set_title(f"{d}D")
        axes[0][j].set_yscale("log")
        axes[1][j].set_xlabel("step")
        if j == 0:
            axes[0][j].set_ylabel("grad norm (smoothed)")
            axes[1][j].set_ylabel("learning rate")
    axes[0][0].legend(frameon=False)
    save(fig, "08_optimization_health")


def plot_convergence(ov):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(DIMS))
    w = 0.38
    for k, m in enumerate(MODELS):
        vals = [ov[(ov.dim == d) & (ov.model == m)]["steps_to_within_5pct"].squeeze()
                if len(ov[(ov.dim == d) & (ov.model == m)]) else np.nan for d in DIMS]
        ax.bar(x + (k - 0.5) * w, vals, w, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}D" for d in DIMS])
    ax.set_ylabel("steps until eval loss within 5% of its minimum")
    ax.set_title("Convergence speed")
    ax.legend(frameon=False)
    save(fig, "09_convergence_speed")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = load_all()
    if not data:
        raise SystemExit("No CSV files found. Check DATA_DIR and file names.")

    print("\nBuilding tables...")
    ov = overall_table(data)
    pt = per_target_table(data)
    st = stats_table(pt)
    ov.to_csv(OUT_DIR / "summary_overall.csv", index=False)
    pt.to_csv(OUT_DIR / "per_target_final.csv", index=False)
    st.to_csv(OUT_DIR / "paired_stats.csv", index=False)

    print("\nBuilding plots...")
    plot_loss_curves(data)
    plot_r2_curves(data)
    plot_gap(data)
    plot_overall_bars(ov)
    plot_scaling(pt)
    plot_delta_heatmap(pt)
    plot_largest_dim(pt)
    plot_optimization(data)
    plot_convergence(ov)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== OVERALL SUMMARY (best-eval checkpoint) ===")
    print(ov[["dim", "model", "n_targets", "best_step", "test_mse", "test_r2",
              "test_r2_lastN_mean", "test_r2_lastN_std", "steps_to_within_5pct"]]
          .round(4).to_string(index=False))
    print("\n=== PAIRED PER-TARGET STATS (PLE - plain, test R2) ===")
    print(st.round(4).to_string(index=False))
    print(f"\nDone. Everything is in ./{OUT_DIR}/")
    print("Reminder: one run per setting -- no seed variance, so interpret small gaps carefully.")


if __name__ == "__main__":
    main()
