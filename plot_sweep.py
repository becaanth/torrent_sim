"""Plot a sweep's summary.csv: x = any swept parameter, y = the same
navigation stats gem_metrics.py compares (time until full dissemination,
total downloaded maps, reachable/contiguous maps), color = strategy.
Works with any sweep produced by run_simulation.py, regardless of what
was actually varied -- just point --x at whichever column you swept. All
three y-metrics are read straight from summary.csv (no event log
needed).

Usage:
    python3 plot_sweep.py results/sequentiality_plus/summary.csv --x radio.fixed.d0
    python3 plot_sweep.py results/base_scenario/summary.csv --x churn.p_bad --color strategy
"""
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Y_METRICS = {
    "finish_time": "Time until full dissemination (finish_time, s)",
    "total_pieces_held": "Total # downloaded maps (all swarms)",
    "reachable_pieces_held": "# reachable maps (contiguous, all swarms)",
}


def plot_sweep(summary_path, x, color="strategy", y_metrics=None, role="TARGET", out="sweep_plot.png"):
    df = pd.read_csv(summary_path)
    df = df[df["role"] == role].copy()
    y_metrics = y_metrics or list(Y_METRICS.keys())

    for col in [x, color, *y_metrics]:
        if col not in df.columns:
            raise ValueError(f"'{col}' is not a column in {summary_path}. "
                              f"Available columns: {list(df.columns)}")

    df = df.dropna(subset=[x, color])  # only these two are needed by every panel
    groups = sorted(df[color].unique())
    cmap = plt.get_cmap("tab10")
    colors = {g: cmap(i % 10) for i, g in enumerate(groups)}

    fig, axes = plt.subplots(1, len(y_metrics), figsize=(6.5 * len(y_metrics), 5.5), squeeze=False)
    axes = axes[0]

    for ax, y in zip(axes, y_metrics):
        # Drop rows missing THIS panel's metric only -- e.g. an
        # incomplete session has no finish_time (so it's absent from
        # that panel) but can still have a real total_pieces_held /
        # reachable_pieces_held (so it belongs in those panels). A
        # shared dropna across all metrics at once would silently
        # discard exactly the failure/partial-progress data this is
        # meant to surface.
        panel_df = df.dropna(subset=[y])
        for g in groups:
            sub = panel_df[panel_df[color] == g].sort_values(x)
            if sub.empty:
                continue
            # individual points (raw, so repeats/variance stay visible)...
            ax.scatter(sub[x], sub[y], color=colors[g], alpha=0.35, s=25, zorder=2)
            # ...and the mean per x-value, connected into a line.
            means = sub.groupby(x)[y].mean().reset_index()
            ax.plot(means[x], means[y], color=colors[g], marker="o", linewidth=2, label=str(g), zorder=3)

        ax.set_xlabel(x)
        ax.set_ylabel(Y_METRICS.get(y, y))
        ax.set_title(Y_METRICS.get(y, y), fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(title=color, fontsize=8)

    fig.suptitle(f"{summary_path}  (role={role}, points=raw, line=mean)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", help="Path to a sweep's summary.csv")
    parser.add_argument("--x", required=True, help="Column to use as the x-axis (the swept parameter)")
    parser.add_argument("--color", default="strategy", help="Column to color/group by (default: strategy)")
    parser.add_argument("--y", nargs="+", default=None,
                         help="Which y-metric(s) to plot (default: both walk_finish_time and finish_time)")
    parser.add_argument("--role", default="TARGET",
                         help="Filter to this role before plotting (default: TARGET -- "
                              "finish_time/total_pieces_held/reachable_pieces_held are only populated on an agent's own target session)")
    parser.add_argument("--out", default="sweep_plot.png")
    args = parser.parse_args()

    plot_sweep(args.summary, args.x, color=args.color, y_metrics=args.y, role=args.role, out=args.out)