"""Robustness vs. Sequentiality scatter (matplotlib), one panel per
strategy scenario, marker size = Throughput, color = swarm.

Switched from Plotly to matplotlib: static PNG output, no browser/Chrome
dependency, standard for paper figures. Combines multiple scenario CSVs
(one per strategy) into a single figure with one subplot per strategy so
they're directly comparable side by side.

Note on plot choice: earlier drafts of this used a ternary plot, but a
ternary plot assumes T+S+R sum to a fixed total (a compositional-data
assumption) which doesn't hold here -- T/S/R are independent metrics, not
shares of a whole. This plain 2D scatter (throughput as marker size
rather than a forced third axis) makes no such assumption. The ternary
version, if still wanted for reference, lives in metrics_plots.py.
"""
import argparse
from pathlib import Path

import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

MARKER_CYCLE = ["o", "^", "s", "D", "v", "P", "X"]


def load_scenario(label, path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["throughput_mbps", "sequentiality", "robustness"]).copy()

    # Exclude seeders explicitly wherever the column says so (current
    # CSV format)...
    if "role" in df.columns:
        df = df[df["role"] != "SEED"]
    # ...and defensively via the same heuristic as before, for
    # older-format CSVs that predate the seed-labeling fix: a seed
    # session always has sequentiality=robustness=0.0 exactly (it never
    # downloads) but nonzero throughput, and adds no information about
    # any actual download/navigation strategy.
    df = df[~((df["sequentiality"] == 0.0) & (df["robustness"] == 0.0))]

    df["strategy"] = label
    return df


def build_swarm_colors(torrent_ids):
    cmap = plt.get_cmap("tab10")
    return {tid: cmap(i % 10) for i, tid in enumerate(sorted(torrent_ids))}


def build_rs_scatter(scenario_files, out_path):
    """scenario_files: list of (label, csv_path) pairs, all plotted
    together on ONE set of axes. x=Sequentiality, y=Robustness, marker
    size=Throughput (Mbps), color=swarm (torrent_id), marker shape=
    strategy scenario."""
    dfs = [load_scenario(label, path) for label, path in scenario_files]
    df = pd.concat(dfs, ignore_index=True)

    strategies = [label for label, _ in scenario_files]
    strategy_markers = {strat: MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, strat in enumerate(strategies)}
    swarm_colors = build_swarm_colors(df["torrent_id"].unique())
    max_throughput = df["throughput_mbps"].max()

    def marker_size(t):
        # area-based scaling (matplotlib's `s` is area, not radius) so
        # size differences read proportionally rather than exaggerated.
        return 30 + 300 * (t / max_throughput)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    for _, row in df.iterrows():
        color = swarm_colors[row["torrent_id"]]
        marker = strategy_markers[row["strategy"]]
        ax.scatter(row["sequentiality"], row["robustness"], s=marker_size(row["throughput_mbps"]),
                   color=color, marker=marker, edgecolors="black", linewidths=0.4,
                   alpha=0.85, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequentiality")
    ax.set_ylabel("Robustness")
    ax.grid(alpha=0.3)

    swarm_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
               markeredgecolor="black", markersize=9, label=f"Swarm {tid}")
        for tid, color in swarm_colors.items()
    ]
    strategy_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=9, label=strat)
        for strat, m in strategy_markers.items()
    ]
    leg1 = ax.legend(handles=swarm_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      fontsize=9, title="Swarm")
    ax.add_artist(leg1)
    ax.legend(handles=strategy_handles, loc="lower left", bbox_to_anchor=(1.02, 0.0),
              fontsize=9, title="Strategy")

    ax.set_title(f"Robustness vs. Sequentiality (marker size = Throughput, max {max_throughput:.2f} Mbps)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def label_from_path(path):
    return Path(path).stem.replace("_", " ").title()


# --------------------------------------------------------------------------
# Total maps/time + reachable maps/time (from sim_log.json, not metrics.csv)
# --------------------------------------------------------------------------

def contiguous_length(piece_set):
    """Same definition as Agent.contiguous_length in torrent_sim.py: how
    many pieces, starting from 0, are held back-to-back with no gap."""
    length = 0
    while length in piece_set:
        length += 1
    return length


def build_maps_over_time(log_path):
    """Replay PIECE_OWNED events into, per (non-seed) agent, a time
    series of (t, total, visitable) where:
      total     = sum over swarms of raw piece count held (dissemination)
      visitable = sum over swarms of contiguous_length held (reachable)
    Seeders are excluded -- they hold everything instantly, which isn't
    the interesting curve here.
    """
    with open(log_path) as f:
        data = json.load(f)
    header, events = data["header"], data["events"]
    agents_by_id = {a["id"]: a for a in header["agents"]}
    seeded_ids = {a["id"] for a in header["agents"] if a["seeded_torrents"]}

    owned = {
        int(aid): {tid: set(pieces) for tid, pieces in per_swarm.items()}
        for aid, per_swarm in header["initial_pieces"].items()
    }
    series = {aid: [] for aid in agents_by_id if aid not in seeded_ids}

    def snapshot(t, agent_id):
        if agent_id not in series:
            return
        total = sum(len(s) for s in owned[agent_id].values())
        visitable = sum(contiguous_length(s) for s in owned[agent_id].values())
        series[agent_id].append((t, total, visitable))

    for aid in series:
        snapshot(0.0, aid)  # t=0 starting point (root piece already owned)

    for e in sorted(events, key=lambda e: e["t"]):
        if e["type"] != "PIECE_OWNED" or e["agent_id"] in seeded_ids:
            continue
        owned[e["agent_id"]][e["torrent_id"]].add(e["piece_id"])
        snapshot(e["t"], e["agent_id"])

    return series


def plot_metric_over_time(scenario_logs, out_path, metric_index, metric_name, title):
    """scenario_logs: list of (label, sim_log.json path) pairs. One line
    per (non-seed) agent, colored by scenario/strategy, all overlaid on
    a single plot. metric_index selects total(1) or visitable(2) from
    build_maps_over_time's (t, total, visitable) tuples."""
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(i % 10) for i, (label, _) in enumerate(scenario_logs)}

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for label, path in scenario_logs:
        series = build_maps_over_time(path)
        for pts in series.values():
            ts = [p[0] for p in pts]
            ys = [p[metric_index] for p in pts]
            ax.step(ts, ys, where="post", color=colors[label], alpha=0.7, linewidth=1.5)

    handles = [Line2D([0], [0], color=c, lw=2, label=label) for label, c in colors.items()]
    ax.legend(handles=handles, title="Strategy", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel(metric_name)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _shade(color, amount, lighten=True):
    """Blend a color toward white (lighten=True) or black (lighten=False)
    by `amount` in [0,1]."""
    c = np.array(mcolors.to_rgb(color))
    target = np.array([1, 1, 1]) if lighten else np.array([0, 0, 0])
    return tuple(c + (target - c) * amount)


def plot_maps_combined(scenario_logs, out_path):
    """Both total (dissemination) and reachable (visitable) curves on one
    plot: color = strategy, shade = metric -- darker line = total, lighter
    line = reachable. Avoids needing two separate figures to compare both
    metrics across strategies at once."""
    cmap = plt.get_cmap("tab10")
    base_colors = {label: cmap(i % 10) for i, (label, _) in enumerate(scenario_logs)}
    dark_colors = {label: _shade(c, 0.0, lighten=False) for label, c in base_colors.items()}  # base itself
    light_colors = {label: _shade(c, 0.55, lighten=True) for label, c in base_colors.items()}

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for label, path in scenario_logs:
        series = build_maps_over_time(path)
        for pts in series.values():
            ts = [p[0] for p in pts]
            totals = [p[1] for p in pts]
            visitables = [p[2] for p in pts]
            ax.step(ts, totals, where="post", color=dark_colors[label], alpha=0.85, linewidth=1.6, zorder=3)
            ax.step(ts, visitables, where="post", color=light_colors[label], alpha=0.85, linewidth=1.6, zorder=2)

    strategy_handles = [Line2D([0], [0], color=c, lw=2.5, label=label) for label, c in dark_colors.items()]
    metric_handles = [
        Line2D([0], [0], color="black", lw=2.5, label="Total (disseminated)"),
        Line2D([0], [0], color=_shade("black", 0.6, lighten=True), lw=2.5, label="Reachable (visitable)"),
    ]
    leg1 = ax.legend(handles=strategy_handles, title="Strategy", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=metric_handles, title="Metric (shade)", loc="upper left", bbox_to_anchor=(1.02, 0.55))

    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Piece count (all swarms)")
    ax.set_title("Map dissemination vs. reachable maps over time\n(darker = total disseminated, lighter = reachable/visitable)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+",
                         default=["rarest_random.csv", "sequential.csv", "cascading.csv"],
                         help="Scenario CSVs for the R-vs-S scatter. Scenario label is "
                              "derived from each filename (e.g. rarest_random.csv -> 'Rarest Random').")
    parser.add_argument("--out", default="rs_scatter.png")
    parser.add_argument("--logs", nargs="+", default=None,
                         help="Scenario sim_log.json files (one per strategy run) for the "
                              "combined total/reachable maps-over-time plot. If omitted, that "
                              "plot is skipped.")
    parser.add_argument("--out-maps", default="maps_over_time_combined.png")
    args = parser.parse_args()

    scenario_files = [(label_from_path(p), p) for p in args.metrics]
    build_rs_scatter(scenario_files, args.out)

    if args.logs:
        scenario_logs = [(label_from_path(p), p) for p in args.logs]
        plot_maps_combined(scenario_logs, args.out_maps)