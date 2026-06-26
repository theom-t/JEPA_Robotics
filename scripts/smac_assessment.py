"""
SMAC Assessment Script — Slice Charts + Convergence Analysis.

Generates one slice chart per hyperparameter showing the marginal
relationship between each HP and the observed SMAC probe score.

Slice chart definition used here (empirical, no surrogate access):
    For each HP: plot (HP value, best-observed cost for that config)
    at the highest budget each config was evaluated at, with a
    polynomial trend line and the incumbent's value marked.

Outputs:
    smac3_output/graphs/
        ├── convergence_trace.png       — optimisation trajectory
        ├── slice_<param>.png           — one file per HP (15 total)
        └── slice_grid_summary.png      — all HPs in one 5×3 grid
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
SWEEP_NAME      = "v1_normalised_probe"
RUNHISTORY_PATH = (
    f"/home/tmainetucker/Repos/JEPA_Robotics/"
    f"smac3_output/{SWEEP_NAME}/0/runhistory.json"
)
OUTPUT_DIR = "/home/tmainetucker/Repos/JEPA_Robotics/smac3_output/graphs"

# ── Filtering ─────────────────────────────────────────────────────────────────
# Any evaluation with cost > COST_CEILING is treated as a pathological outlier
# and excluded from all charts. Config 2 (cost 1.24) is the current offender.
# The normalised probe score has a natural upper bound of ~1.0 for a model that
# is barely better than random on all three dimensions, so 1.0 is a sound ceil.
COST_CEILING: float = 1.0

# ── Hyperparameter metadata ───────────────────────────────────────────────────
# (name, is_log_scale, is_categorical)
HP_META = [
    ("activation_fn",       False, True ),
    ("batch_size",          False, True ),
    ("latent_dim",          False, True ),
    ("learning_rate",       True,  False),
    ("loss_alpha",          False, False),
    ("masking_ratio",       False, False),
    ("num_heads",           False, True ),
    ("patch_size",          False, True ),
    ("probe_learning_rate", True,  False),
    ("seq_len",             False, False),
    ("sigreg_weight",       True,  False),
    ("tau",                 False, False),
    ("vit_depth",           False, False),
    ("weight_decay",        True,  False),
    ("wm_depth",            False, False),
]

# Budget → colour for scatter points
BUDGET_COLORS = {4.0: "#5B9BD5", 8.0: "#ED7D31", 16.0: "#70AD47"}

# Origin → marker
ORIGIN_MARKERS = {
    "Initial Design: Sobol":                   "o",
    "Acquisition Function Maximizer: Local Search": "D",
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         120,
})


def load_runhistory(path: str) -> dict:
    """Load and parse the SMAC runhistory.json."""
    with open(path) as f:
        return json.load(f)


def best_per_config(data: list, configs: dict, origins: dict) -> list:
    """
    For each unique config, keep only the entry at its highest observed budget.
    Evaluations with cost > COST_CEILING are silently dropped as outliers.
    Returns a list of dicts with keys:
        config_id, budget, cost, config, origin
    """
    best: dict = {}
    for entry in data:
        cid   = str(entry["config_id"])
        cost  = entry["cost"]
        budg  = entry["budget"]
        if cost > COST_CEILING:
            continue  # outlier — skip entirely
        if cid not in best or budg > best[cid]["budget"] or (
            budg == best[cid]["budget"] and cost < best[cid]["cost"]
        ):
            best[cid] = {
                "config_id": cid,
                "budget":    budg,
                "cost":      cost,
                "config":    configs.get(cid, {}),
                "origin":    origins.get(cid, "Unknown"),
            }
    return list(best.values())


def find_incumbent(records: list) -> dict:
    """Return the record with the lowest cost."""
    return min(records, key=lambda r: r["cost"])


def smooth_trend(x: np.ndarray, y: np.ndarray, log_x: bool = False, degree: int = 2):
    """
    Fit a polynomial trend to (x, y) and return (x_line, y_line).
    Fits in log-x space when log_x=True.
    Returns None, None if fewer than 3 points.
    """
    if len(x) < 3:
        return None, None
    x_fit = np.log10(x) if log_x else x.copy()
    # Clip y to reasonable range to avoid exploding fits
    y_fit = np.clip(y, 0.0, 2.0)
    try:
        coeffs = np.polyfit(x_fit, y_fit, degree)
        x_lin  = np.linspace(x_fit.min(), x_fit.max(), 200)
        y_lin  = np.polyval(coeffs, x_lin)
        if log_x:
            x_lin = 10 ** x_lin
        return x_lin, y_lin
    except Exception:
        return None, None


def make_slice_continuous(
    ax: plt.Axes,
    records: list,
    hp_name: str,
    log_scale: bool,
    incumbent: dict,
) -> None:
    """Draw a continuous-variable slice chart onto ax."""
    groups = defaultdict(list)
    for r in records:
        val = r["config"].get(hp_name)
        if val is None:
            continue
        groups[(r["budget"], r["origin"])].append((val, r["cost"]))

    all_x, all_y = [], []

    for (budget, origin), points in sorted(groups.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        all_x.extend(xs)
        all_y.extend(ys)
        color  = BUDGET_COLORS.get(budget, "#999999")
        marker = ORIGIN_MARKERS.get(origin, "s")
        ax.scatter(
            xs, ys,
            color=color, marker=marker,
            s=55, alpha=0.85, zorder=3,
            edgecolors="white", linewidths=0.4,
        )

    # Trend line over all points
    if all_x:
        xs_arr = np.array(all_x, dtype=float)
        ys_arr = np.array(all_y, dtype=float)
        sort_idx = np.argsort(xs_arr)
        x_t, y_t = smooth_trend(xs_arr[sort_idx], ys_arr[sort_idx], log_x=log_scale)
        if x_t is not None:
            ax.plot(x_t, y_t, color="#C00000", linewidth=1.6,
                    linestyle="--", alpha=0.75, zorder=2, label="Trend")

    # Incumbent value marker
    inc_val = incumbent["config"].get(hp_name)
    if inc_val is not None:
        ax.axvline(inc_val, color="#FFD700", linewidth=2.0,
                   linestyle="-", zorder=4, label=f"Incumbent ({inc_val:.4g})")
        ax.scatter([inc_val], [incumbent["cost"]],
                   color="#FFD700", marker="*", s=220, zorder=5,
                   edgecolors="#7F6000", linewidths=0.8)

    if log_scale:
        ax.set_xscale("log")

    # Clip y-axis so outliers outside COST_CEILING don't compress the view
    if all_y:
        y_lo = max(0.0, min(all_y) * 0.97)
        y_hi = min(COST_CEILING, max(all_y) * 1.03)
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(hp_name, fontsize=8)
    ax.set_ylabel("SMAC Score (↓ better)", fontsize=8)
    ax.set_title(f"Slice: {hp_name}", fontsize=9, fontweight="bold")


def make_slice_categorical(
    ax: plt.Axes,
    records: list,
    hp_name: str,
    incumbent: dict,
) -> None:
    """Draw a categorical-variable slice chart (strip + mean bar) onto ax."""
    cat_data = defaultdict(list)
    cat_budgets = defaultdict(list)
    cat_origins = defaultdict(list)

    for r in records:
        val = r["config"].get(hp_name)
        if val is None:
            continue
        key = str(val)
        cat_data[key].append(r["cost"])
        cat_budgets[key].append(r["budget"])
        cat_origins[key].append(r["origin"])

    categories = sorted(cat_data.keys(),
                        key=lambda x: (float(x) if x.replace('.','').isdigit() else x))

    for ci, cat in enumerate(categories):
        costs   = cat_data[cat]
        budgets = cat_budgets[cat]
        origins = cat_origins[cat]
        n = len(costs)

        # Jitter x positions
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, n)
        xs = np.full(n, ci) + jitter

        for x, cost, budget, origin in zip(xs, costs, budgets, origins):
            color  = BUDGET_COLORS.get(budget, "#999999")
            marker = ORIGIN_MARKERS.get(origin, "s")
            ax.scatter([x], [cost], color=color, marker=marker,
                       s=55, alpha=0.85, zorder=3,
                       edgecolors="white", linewidths=0.4)

        # Mean bar
        mean_val = np.mean(costs)
        ax.hlines(mean_val, ci - 0.3, ci + 0.3,
                  colors="#C00000", linewidths=2.0, zorder=4)

    # Highlight incumbent category
    inc_val = incumbent["config"].get(hp_name)
    if inc_val is not None:
        inc_key = str(inc_val)
        if inc_key in categories:
            ci = categories.index(inc_key)
            ax.axvspan(ci - 0.4, ci + 0.4,
                       color="#FFD700", alpha=0.18, zorder=1)
            ax.scatter([ci], [incumbent["cost"]],
                       color="#FFD700", marker="*", s=220, zorder=5,
                       edgecolors="#7F6000", linewidths=0.8)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=7, rotation=15)

    # Clip y-axis
    all_cat_costs = [c for vals in cat_data.values() for c in vals]
    if all_cat_costs:
        y_lo = max(0.0, min(all_cat_costs) * 0.97)
        y_hi = min(COST_CEILING, max(all_cat_costs) * 1.03)
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(hp_name, fontsize=8)
    ax.set_ylabel("SMAC Score (↓ better)", fontsize=8)
    ax.set_title(f"Slice: {hp_name}", fontsize=9, fontweight="bold")


def make_convergence_trace(data: list, configs: dict, origins: dict, output_dir: str) -> None:
    """Plot the optimisation trajectory over all trials."""
    fig, ax = plt.subplots(figsize=(12, 5))

    trial_num, costs, best_costs = [], [], []
    current_best = float("inf")
    skipped = 0

    for i, entry in enumerate(data):
        cost = entry["cost"]
        budg = entry["budget"]
        cid  = str(entry["config_id"])
        orig = origins.get(cid, "Unknown")

        # Track best ignoring ceiling filter so the red line is always correct
        if cost < current_best:
            current_best = cost

        # Exclude outliers from the scatter but not from the best-cost line
        if cost > COST_CEILING:
            skipped += 1
            trial_num.append(i + 1)
            costs.append(None)          # gap in scatter
            best_costs.append(current_best)
            continue

        color  = BUDGET_COLORS.get(budg, "#999999")
        marker = ORIGIN_MARKERS.get(orig, "s")
        ax.scatter([i + 1], [cost], color=color, marker=marker,
                   s=50, alpha=0.8, zorder=3)

        trial_num.append(i + 1)
        costs.append(cost)
        best_costs.append(current_best)

    if skipped:
        print(f"  (convergence trace: {skipped} outlier(s) above "
              f"COST_CEILING={COST_CEILING} hidden from scatter)")

    ax.plot(trial_num, best_costs, color="#C00000",
            linewidth=2.2, label="Best cost so far", zorder=4)

    # Keep y-axis tightly around the visible range
    visible = [c for c in costs if c is not None]
    if visible:
        ax.set_ylim(
            max(0.0, min(visible) * 0.97),
            min(COST_CEILING, max(visible) * 1.03),
        )

    # Legend
    budget_handles = [
        mpatches.Patch(color=c, label=f"Budget={b:.0f}ep")
        for b, c in BUDGET_COLORS.items()
    ]
    origin_handles = [
        Line2D([0], [0], marker=m, color="grey", label=o.split(":")[0],
               markersize=7, linestyle="None")
        for o, m in ORIGIN_MARKERS.items()
    ]
    best_handle = Line2D([0], [0], color="#C00000", linewidth=2, label="Best so far")
    ax.legend(handles=budget_handles + origin_handles + [best_handle],
              fontsize=7, ncol=3, loc="upper right")

    ax.set_xlabel("Trial number")
    ax.set_ylabel("SMAC Score (↓ better)")
    ax.set_title(f"SMAC Optimisation Convergence — {SWEEP_NAME}", fontweight="bold")

    path = os.path.join(output_dir, "convergence_trace.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ convergence_trace.png")


def make_individual_slice(
    records: list,
    hp_name: str,
    log_scale: bool,
    is_categorical: bool,
    incumbent: dict,
    output_dir: str,
) -> None:
    """Save a standalone slice chart PNG for one hyperparameter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if is_categorical:
        make_slice_categorical(ax, records, hp_name, incumbent)
    else:
        make_slice_continuous(ax, records, hp_name, log_scale, incumbent)

    path = os.path.join(output_dir, f"slice_{hp_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ slice_{hp_name}.png")


def make_summary_grid(
    records: list,
    incumbent: dict,
    output_dir: str,
) -> None:
    """Draw all 15 slice charts in a 5×3 summary grid."""
    COLS, ROWS = 3, 5
    fig, axes = plt.subplots(ROWS, COLS, figsize=(18, 28))
    axes_flat = axes.flatten()

    for idx, (hp_name, log_scale, is_cat) in enumerate(HP_META):
        ax = axes_flat[idx]
        if is_cat:
            make_slice_categorical(ax, records, hp_name, incumbent)
        else:
            make_slice_continuous(ax, records, hp_name, log_scale, incumbent)

    # Hide any unused axes (15 HPs, 15 cells → none to hide in this case)
    for idx in range(len(HP_META), ROWS * COLS):
        axes_flat[idx].set_visible(False)

    # Shared legend at top
    budget_handles = [
        mpatches.Patch(color=c, label=f"Budget {b:.0f}ep")
        for b, c in BUDGET_COLORS.items()
    ]
    origin_handles = [
        Line2D([0], [0], marker=m, color="grey", label=o.split(":")[0],
               markersize=8, linestyle="None")
        for o, m in ORIGIN_MARKERS.items()
    ]
    star_handle = Line2D([0], [0], marker="*", color="#FFD700",
                         markersize=12, linestyle="None",
                         markeredgecolor="#7F6000", label="Incumbent")
    trend_handle = Line2D([0], [0], color="#C00000", linewidth=1.6,
                          linestyle="--", label="Polynomial trend")
    fig.legend(
        handles=budget_handles + origin_handles + [star_handle, trend_handle],
        loc="upper center", ncol=7, fontsize=9,
        bbox_to_anchor=(0.5, 1.005),
    )

    fig.suptitle(
        f"SMAC Slice Charts — {SWEEP_NAME}\n"
        f"Incumbent: Config {incumbent['config_id']} "
        f"| Score {incumbent['cost']:.4f} "
        f"@ budget {incumbent['budget']:.0f}ep",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "slice_grid_summary.png")
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  ✔ slice_grid_summary.png")


def main() -> None:
    if not os.path.exists(RUNHISTORY_PATH):
        print(f"[ERROR] Runhistory not found: {RUNHISTORY_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading runhistory from:\n  {RUNHISTORY_PATH}\n")

    history = load_runhistory(RUNHISTORY_PATH)
    data    = history.get("data", [])
    configs = history.get("configs", {})
    origins = history.get("config_origins", {})

    print(f"Total trial evaluations : {len(data)}")

    records   = best_per_config(data, configs, origins)
    incumbent = find_incumbent(records)

    print(f"Unique configs evaluated: {len(records)}")
    print(f"Current incumbent        : Config {incumbent['config_id']}"
          f" | Score {incumbent['cost']:.4f}"
          f" @ budget {incumbent['budget']:.0f}ep")
    print(f"\nGenerating charts → {OUTPUT_DIR}\n")

    # 1. Convergence trace
    make_convergence_trace(data, configs, origins, OUTPUT_DIR)

    # 2. Individual slice per HP
    for hp_name, log_scale, is_cat in HP_META:
        make_individual_slice(records, hp_name, log_scale, is_cat, incumbent, OUTPUT_DIR)

    # 3. Combined summary grid
    make_summary_grid(records, incumbent, OUTPUT_DIR)

    print(f"\n✅ {1 + len(HP_META) + 1} charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
