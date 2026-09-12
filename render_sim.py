"""Replay a torrent_sim.py run into a rendered animation (MP4/GIF).

Consumes sim_log.json (header + timestamped event list) produced by
torrent_sim.py's __main__ block. Never touches the live simulation objects
-- everything needed to reconstruct world state at any moment is either in
the header (static setup) or reconstructed by replaying events in order.

World-state reconstruction:
  - positions: start at header start_position, updated by POSITION events.
  - piece ownership (for the per-swarm bitmaps): starts from
    initial_pieces, updated by PIECE_OWNED events.
  - active transfers (for the leech-link arrows): a transfer is active
    from its TRANSFER_START until its TRANSFER_END.

Playback uses compressed time: the sim's real timeline has long idle
gaps (waiting on seeder ticks) followed by bursts of transfer activity,
so mapping 1:1 to wall-clock video time would look like nothing-then-
everything. Instead the total sim timespan is stretched/compressed evenly
across a fixed video duration.
"""
import json
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

STRATEGY_COLORS = {
    "rarest_random": "#1f77b4",
    "sequential": "#2ca02c",
    "cascading": "#d62728",
    "hybrid": "#9467bd",
    "segment_random": "#ff7f0e",
}
SEED_COLOR = "#000000"


def load_log(path):
    with open(path) as f:
        return json.load(f)


def build_frame_times(events, n_frames):
    """Evenly spaced sim-times from 0 to the log's last event, regardless
    of how bursty real activity is (see module docstring)."""
    max_t = max((e["t"] for e in events), default=0.0)
    if n_frames <= 1 or max_t == 0:
        return [max_t]
    return [max_t * i / (n_frames - 1) for i in range(n_frames)]


class ReplayState:
    """Incrementally applies events up to a target time, keeping a
    running world state. Call advance_to(t) with non-decreasing t values
    (one call per rendered frame) -- it only walks forward through the
    event list, never rescans from the start."""

    def __init__(self, header):
        self.header = header
        self.positions = {
            a["id"]: tuple(a["start_position"]) for a in header["agents"]
        }
        self.owned = {
            int(agent_id): {tid: set(pieces) for tid, pieces in per_swarm.items()}
            for agent_id, per_swarm in header["initial_pieces"].items()
        }
        self.active_transfers = {}  # transfer_id -> (torrent_id, downloader_id, uploader_id)
        self.horizons = {tid: s["horizon_D"] or 1 for tid, s in header["swarms"].items()}
        self.finalized = set()

    def apply(self, event):
        et = event["type"]
        if et == "POSITION":
            self.positions[event["agent_id"]] = (event["x"], event["y"])
        elif et == "PIECE_OWNED":
            self.owned[event["agent_id"]][event["torrent_id"]].add(event["piece_id"])
        elif et == "TRANSFER_START":
            self.active_transfers[event["transfer_id"]] = (
                event["torrent_id"], event["downloader_id"], event["uploader_id"]
            )
        elif et == "TRANSFER_END":
            self.active_transfers.pop(event["transfer_id"], None)
        elif et == "APPEND":
            self.horizons[event["torrent_id"]] = event["horizon"]
        elif et == "FINALIZE":
            self.finalized.add(event["torrent_id"])


def agent_style(agent_meta):
    """Color/marker for one agent, by role/strategy."""
    if agent_meta["seeded_torrents"]:
        return SEED_COLOR, "s"  # square marker for seeders
    strat = agent_meta["strategies"].get(agent_meta["target_torrent_id"], "rarest_random")
    return STRATEGY_COLORS.get(strat, "#7f7f7f"), "o"


def render(log_path, out_path, n_frames=150, fps=15):
    data = load_log(log_path)
    header, events = data["header"], data["events"]
    events = sorted(events, key=lambda e: e["t"])
    agents_by_id = {a["id"]: a for a in header["agents"]}
    swarms = header["swarms"]

    frame_times = build_frame_times(events, n_frames)
    state = ReplayState(header)

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])
    ax_world = fig.add_subplot(grid[:, 0])
    bitmap_axes = {
        tid: fig.add_subplot(grid[i // 2, 1 + i % 2])
        for i, tid in enumerate(swarms)
    }

    event_idx = 0

    def draw_frame(frame_i):
        nonlocal event_idx
        t = frame_times[frame_i]
        # Advance world state through every event up to this frame's time.
        while event_idx < len(events) and events[event_idx]["t"] <= t:
            state.apply(events[event_idx])
            event_idx += 1

        # --- world panel ---
        ax_world.clear()
        max_horizon = max(state.horizons.values())
        pad = max_horizon * 1.15 + 1
        ax_world.set_xlim(-pad, pad)
        ax_world.set_ylim(-pad, pad)
        ax_world.set_aspect("equal")
        ax_world.set_title(f"t = {t:5.1f}s")

        for tid, info in swarms.items():
            dx, dy = info["direction"]
            horizon = state.horizons[tid]
            ax_world.plot([0, dx * horizon], [0, dy * horizon], "--", color="lightgray", zorder=0)
            style = "solid" if tid in state.finalized else "dotted"
            ax_world.plot([0, dx * horizon], [0, dy * horizon], linestyle=style,
                          color="gray", linewidth=0.5, zorder=0)

        # active leech links: uploader -> downloader
        for tid, downloader_id, uploader_id in state.active_transfers.values():
            ux, uy = state.positions[uploader_id]
            dx_, dy_ = state.positions[downloader_id]
            ax_world.annotate("", xy=(dx_, dy_), xytext=(ux, uy),
                               arrowprops=dict(arrowstyle="->", color="orange", alpha=0.6, lw=1.2))

        for agent_id, pos in state.positions.items():
            color, marker = agent_style(agents_by_id[agent_id])
            ax_world.scatter(*pos, color=color, marker=marker, s=80, zorder=3, edgecolors="black", linewidths=0.5)
            ax_world.annotate(str(agent_id), pos, fontsize=7, xytext=(3, 3), textcoords="offset points")

        # --- bitmap panels: one per swarm ---
        for tid, ax in bitmap_axes.items():
            ax.clear()
            horizon = state.horizons[tid]
            participant_ids = sorted(
                aid for aid, per_swarm in state.owned.items() if tid in per_swarm
            )
            grid_data = [
                [1 if p in state.owned[aid][tid] else 0 for p in range(horizon)]
                for aid in participant_ids
            ]
            ax.imshow(grid_data, aspect="auto", cmap="Greens", vmin=0, vmax=1)
            ax.set_yticks(range(len(participant_ids)))
            ax.set_yticklabels([str(a) for a in participant_ids], fontsize=6)
            ax.set_xlabel("piece", fontsize=7)
            ax.set_title(f"Swarm {tid}", fontsize=9)

        fig.tight_layout()
        return []

    writer_cls, ext = (FFMpegWriter, "mp4") if _ffmpeg_available() else (PillowWriter, "gif")
    if not out_path.endswith(f".{ext}"):
        out_path = out_path.rsplit(".", 1)[0] + f".{ext}"

    anim = FuncAnimation(fig, draw_frame, frames=len(frame_times), blit=False)
    anim.save(out_path, writer=writer_cls(fps=fps))
    plt.close(fig)
    return out_path


def _ffmpeg_available():
    import shutil
    return shutil.which("ffmpeg") is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="sim_log.json")
    parser.add_argument("--out", default="sim_render.mp4")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    out = render(args.log, args.out, n_frames=args.frames, fps=args.fps)
    print(f"Wrote {out}")