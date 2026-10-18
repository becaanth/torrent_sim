"""Download progress over time for a sweep, from each trial's sim_log.json.

One axes (or one panel per policy with --facet): x = time, y = # maps held
in the agent's own target swarm. The seeder's curve (maps published so far)
is the diagonal envelope -- nobody can hold a map before it exists, so the
vertical gap under it is how far behind a peer is.

Encoding:
  colour hue   = policy (--color-by, default agents.peers.0.strategy)
  colour shade = sweep value (--tint-by), a DISCRETE ladder of at most
                 --max-traces (default 6) shades, light = low value, dark = high.
                 More values than that are thinned evenly (ends kept) and the
                 choice is printed; --tint-values picks them explicitly.
  text label   = the sweep value, printed on its own line (staggered along
                 the line so close-together lines don't collide).
                 Lines that coincide are merged into one label ("200-4000").
  solid line   = mean peer, EXCLUDING peers flagged slow (is_slow) -- so the
                 heterogeneity sweeps show how the healthy fleet fares, not
                 the degraded radio itself. With no slow peers it is simply
                 the mean over all peers. Averaged over repeats.
  dashed colour line = (multi-swarm runs) the peers' OTHER swarms, same colour as the solid
                 target-swarm line, one dashed line per other swarm, unlabelled
  shaded band  = (--spread repeats|peers) min-max behind each solid line:
                 repeats = min/max of the per-trial mean peer across repeats
                           (seed noise);
                 peers   = min/max over individual healthy peers, pooled over
                           all repeats (what the mean hides).
  thick black  = seeder (drawn beneath the traces so a trace lying on it stays visible)

Needs the sweep to have been run with `write_logs: true` (reads
{sweep_dir}/trial_XXXXX/sim_log.json plus {sweep_dir}/manifest.jsonl).

Usage:
    python3 plot_progress.py results/fixed_d0_base_convoy --tint-by radio.fixed.d0
    python3 plot_progress.py results/fixed_d0_base_convoy --tint-by radio.fixed.d0 --facet --out prog.png
"""
import argparse
import json
import os
from collections import defaultdict

import colorsys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Same policy hues as metrics_plots.py so figures stay consistent.
# STRATEGY_COLORS = {
#     "rarest_random": "#ff7f0e",
#     "sequential": "#2ca02c",
#     "cascading": "#1f77b4",
#     "hybrid": "#9467bd",
#     "segment_random": "#d62728",
# }
STRATEGY_COLORS = {
    "rarest_random": "#1f77b4",
    "sequential": "#2ca02c",
    "cascading": "#d62728",
    "hybrid": "#9467bd",
    "segment_random": "#d62728",
}
L_LIGHT, L_DARK = 0.78, 0.24   # HLS lightness of the lightest / darkest shade in a ladder
MIN_SAT = 0.55                 # keep light shades chromatic instead of washing out to grey


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_manifest(sweep_dir):
    path = os.path.join(sweep_dir, "manifest.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- point this at a sweep output directory")
    trials = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            trials[rec["trial_index"]] = rec["overrides"]
    return trials


def agent_curve(initial_pieces, owned):
    """Step function (times, maps, reachable) for one agent in one swarm.
    `owned` is [(t, piece_id), ...] in log order. reachable = contiguous run
    of held pieces from piece 0, advanced incrementally (O(1) per piece)."""
    held = set(initial_pieces)
    contig = 0
    while contig in held:
        contig += 1
    times, maps, reach = [0.0], [len(held)], [contig]
    for t, p in owned:
        if p in held:
            continue
        held.add(p)
        if p == contig:
            while contig in held:
                contig += 1
        times.append(t)
        maps.append(len(held))
        reach.append(contig)
    return np.array(times), np.array(maps, dtype=float), np.array(reach, dtype=float)


def cascade_chain(header, links, seeder_id, peer_ids, torrent_id):
    """Upstream-to-downstream agent order of ONE swarm's cascade, rebuilt from who
    actually uploaded to whom (a cascading peer only ever downloads from its one
    upstream). `links` = {uploader_id: {downloader_ids}} for that swarm. None if
    the peers aren't all cascading in it or the links don't form a single chain."""
    strat = {a["id"]: a["strategies"].get(torrent_id) for a in header["agents"]}
    if not peer_ids or any(strat[p] != "cascading" for p in peer_ids):
        return None
    chain, cur = [], seeder_id
    while cur in links and len(links[cur]) == 1:
        cur = next(iter(links[cur]))
        chain.append(cur)
    return chain if sorted(chain) == sorted(peer_ids) else None


def load_trial_curves(log_path):
    """-> dict(seeder=(times, maps, reach), peers=[(times, maps, reach, is_slow)], t_end)"""
    with open(log_path) as f:
        log = json.load(f)
    header, events = log["header"], log["events"]
    target = {a["id"]: a["target_torrent_id"] for a in header["agents"]}
    seeded = {a["id"]: set(a["seeded_torrents"]) for a in header["agents"]}
    is_slow = {a["id"]: bool(a.get("is_slow", False)) for a in header["agents"]}
    initial = {int(k): v for k, v in header["initial_pieces"].items()}

    owned = defaultdict(list)
    links = defaultdict(lambda: defaultdict(set))     # swarm -> uploader -> {downloaders}
    t_end = 0.0
    for e in events:
        if e["type"] == "TRANSFER_START":
            links[e["torrent_id"]][e["uploader_id"]].add(e["downloader_id"])
        elif e["type"] == "PIECE_OWNED":
            aid = e["agent_id"]
            owned[(aid, e["torrent_id"])].append((e["t"], e["piece_id"]))
            t_end = max(t_end, e["t"])

    strategies = {a["id"]: a["strategies"] for a in header["agents"]}
    seeders, peers = {}, []          # seeders: swarm -> (agent id, curve)
    for aid in target:
        tid = target[aid]
        if tid in seeded[aid]:
            seeders[tid] = (aid, agent_curve(initial[aid].get(tid, []), owned[(aid, tid)]))
        else:
            curve = agent_curve(initial[aid].get(tid, []), owned[(aid, tid)])
            # the peer's other swarms (it relays them); {} in a single-swarm run
            others = {o: agent_curve(initial[aid].get(o, []), owned[(aid, o)])
                      for o in strategies[aid] if o != tid and o not in seeded[aid]}
            peers.append((*curve, is_slow[aid], aid, others, tid))
    if not seeders or not peers:
        raise ValueError(f"{log_path}: need at least one seeder and one peer")
    ptid = peers[0][6]               # the swarm the peers are travelling in
    if ptid not in seeders:
        raise ValueError(f"{log_path}: no seeder for the peers' target swarm '{ptid}'")
    chains = {tid: cascade_chain(header, links[tid], sid, [p[4] for p in peers if tid in strategies[p[4]]], tid)
              for tid, (sid, _) in seeders.items()}
    return {"seeder": seeders[ptid][1], "peers": peers, "t_end": t_end, "chains": chains}


def hold(times, vals, grid):
    """Value of a step function at each grid time (last record at or before it)."""
    idx = np.searchsorted(times, grid, side="right") - 1
    return vals[np.clip(idx, 0, None)]


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------

def mix(rgb, other, w):
    return tuple((1 - w) * c + w * o for c, o in zip(rgb, other))


def shade(base, k, n):
    """k-th of n discrete shades of `base` (0 = lightest, n-1 = darkest),
    evenly spaced in HLS lightness so neighbouring steps look equally far apart."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(base))
    lightness = l if n == 1 else L_LIGHT - (L_LIGHT - L_DARK) * k / (n - 1)
    return colorsys.hls_to_rgb(h, lightness, max(s, MIN_SAT))


def select_values(values, max_traces, explicit=None):
    """Which sweep values get a trace. Explicit list wins; otherwise thin an
    over-long sweep evenly by rank, always keeping the first and last."""
    vals = sorted(set(values), key=lambda v: (isinstance(v, str), v))
    if explicit is not None:
        keep = [v for v in vals if fmt_val(v) in {fmt_val(e) for e in explicit}]
        if not keep:
            raise ValueError(f"--tint-values matched nothing; available: {[fmt_val(v) for v in vals]}")
        return keep
    if len(vals) <= max_traces:
        return vals
    idx = sorted({int(round(i)) for i in np.linspace(0, len(vals) - 1, max_traces)})
    keep = [vals[i] for i in idx]
    print(f"note: {len(vals)} sweep values > --max-traces {max_traces}; showing "
          f"{[fmt_val(v) for v in keep]} (use --tint-values to choose)")
    return keep


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def fmt_val(v):
    return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def norm_pts(grid, curve, t_max, ymax):
    """Curve as points in unit-square plot space, so distances mean 'how far
    apart do these look on the page' regardless of axis units."""
    return np.c_[grid / t_max, curve / ymax]


def curve_dist(pa, pb):
    """Symmetric Hausdorff distance between two curves in plot space."""
    d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
    return max(d.min(1).max(), d.min(0).max())


def cluster_cells(tint_vals, pts, tol):
    """Merge neighbouring sweep values (in sorted order) whose curves are
    visually indistinguishable (within `tol` of the cluster's first curve, in
    unit plot space), so one label serves them all."""
    clusters = []
    for v in tint_vals:
        if clusters and curve_dist(pts[v], pts[clusters[-1][0]]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def place_labels(fig, ax, labels, grid, ymax, min_sep=0.03):
    """Inline labels: each label is centred ON its own line. Pass 1 only
    accepts spots at least `min_sep` (unit plot space) away from every OTHER
    label's line, so a label can't sit on a segment shared by several lines;
    pass 2 relaxes that if nothing qualifies. Either way a spot must stay inside
    the axes and not collide with a label already placed. Finishing lines are
    scanned from the top of their rise downward, unfinished lines from the
    right edge leftward. White backing keeps text readable over crossings."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer)
    t_max = grid[-1]
    placed = []

    def candidates(lab):
        c = lab["curve"]
        if lab["kind"] == "fin":
            for frac in np.arange(0.92, 0.04, -0.04):
                y = frac * ymax
                yield grid[int(np.argmax(c >= y))], y
        else:
            for frac in np.arange(0.94, 0.05, -0.05):
                i = int(frac * (len(grid) - 1))
                yield grid[i], float(c[i])

    def sep_ok(lab, cx, cy):
        px, py = cx / t_max, cy / ymax
        return all(np.hypot(o[:, 0] - px, o[:, 1] - py).min() >= min_sep for o in lab["others"])

    def try_place(lab, strict):
        for cx, cy in candidates(lab):
            if strict and not sep_ok(lab, cx, cy):
                continue
            t = ax.text(cx, cy, lab["text"], ha="center", va="center", fontsize=7,
                        color=lab["color"], zorder=6,
                        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
            bb = mtext.Text.get_window_extent(t, renderer).expanded(1.12, 1.25)
            inside = bb.x0 >= axbb.x0 and bb.x1 <= axbb.x1 and bb.y0 >= axbb.y0 and bb.y1 <= axbb.y1
            if inside and not any(bb.overlaps(p) for p in placed):
                placed.append(bb)
                return True
            t.remove()
        return False

    for lab in sorted(labels, key=lambda l: l["order"]):
        if not (try_place(lab, True) or try_place(lab, False)):
            cx, cy = next(candidates(lab))   # nowhere free: keep a label rather than drop it
            ax.text(cx, cy, lab["text"], ha="center", va="center", fontsize=7, color=lab["color"], zorder=6)


def plot_progress(sweep_dir, color_by, tint_by, out=None, t_max=None, facet=False,
                  max_traces=6, tint_values=None, split_chain=False, spread="none",
                  n_grid=600, ax=None, labels_out=None, legend=True):
    """Standalone: draws and saves its own figure to `out`.
    Embedded (see plot_progress_row): pass `ax` to draw into a panel of the
    caller's figure; the inline value labels need the final layout, so they are
    appended to `labels_out` as (ax, labels, grid, ymax, {policy: colour}, has_other_swarms, has_upstream) for the caller to place
    once the figure is laid out. No saving or layout is done in that mode."""
    if ax is not None and (facet or labels_out is None):
        raise ValueError("embedding via ax= needs facet=False and a labels_out list")
    if ax is None and out is None:
        raise ValueError("pass out= (standalone) or ax= (embedded)")
    trials = load_manifest(sweep_dir)
    for key in (color_by, tint_by):
        sample = next(iter(trials.values()))
        if key not in sample:
            raise ValueError(f"'{key}' is not a swept/overridden key. Available: {sorted(sample)}")

    keep = set(select_values([o[tint_by] for o in trials.values()], max_traces, tint_values))
    trials = {i: o for i, o in trials.items() if o[tint_by] in keep}

    raw = {}
    for idx in sorted(trials):
        log_path = os.path.join(sweep_dir, f"trial_{idx:05d}", "sim_log.json")
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"{log_path} missing -- re-run the sweep with write_logs: true")
        raw[idx] = load_trial_curves(log_path)

    t_max = t_max or max(r["t_end"] for r in raw.values())
    grid = np.linspace(0.0, t_max, n_grid)

    # (color value, tint value) -> per-trial mean-peer curve on the common grid
    groups = defaultdict(list)       # main (solid) line: mean healthy peer, or healthy peers DOWNSTREAM of a slow node
    other_groups = defaultdict(list)  # (policy, tint, swarm) -> mean healthy peer in a NON-target swarm (dashed)
    pool = defaultdict(list)         # (policy, tint) -> (n_peers x grid) arrays of the peers behind the solid line
    up_groups = defaultdict(list)    # --split-chain only: healthy peers UPSTREAM of the slow node (dotted)
    seeder_curve = None
    n_skipped = n_split = 0
    for idx, r in raw.items():
        key = (trials[idx][color_by], trials[idx][tint_by])
        peers = r["peers"]
        healthy = [hold(p[0], p[1], grid) for p in peers if not p[3]]
        if not healthy:           # every peer slow: no healthy mean to draw
            n_skipped += 1
            continue
        def side(p, tid):
            """'up' / 'dn': is peer p upstream / downstream of the slow peer in swarm `tid`'s cascade
            chain (None if that swarm has no valid chain or no slow peer in it)."""
            chain = r["chains"].get(tid)
            if not chain:
                return None
            pos = {a: k for k, a in enumerate(chain)}
            slow_pos = [pos[q[4]] for q in peers if q[3] and q[4] in pos]
            if not slow_pos or p[4] not in pos or pos[p[4]] == min(slow_pos):
                return None
            return "up" if pos[p[4]] < min(slow_pos) else "dn"
        # each peer's TARGET curve is split by the chain of its OWN target swarm
        sides = {p[4]: side(p, p[6]) for p in peers if not p[3]}
        do_split = split_chain and any(p[3] for p in peers) and all(s for s in sides.values())
        if do_split:
            up = [hold(p[0], p[1], grid) for p in peers if not p[3] and sides[p[4]] == "up"]
            dn = [hold(p[0], p[1], grid) for p in peers if not p[3] and sides[p[4]] == "dn"]
            if up:
                up_groups[key].append(np.vstack(up).mean(axis=0))
            if dn:
                groups[key].append(np.vstack(dn).mean(axis=0))
                pool[key].append(np.vstack(dn))
            n_split += 1
        else:
            groups[key].append(np.vstack(healthy).mean(axis=0))
            pool[key].append(np.vstack(healthy))
        for o in sorted({o for p in peers if not p[3] for o in p[5]}):
            # other swarms follow the same rule as the solid line: with a split, only the peers
            # downstream of the slow peer in THAT swarm's chain; otherwise every healthy peer
            sel = [p for p in peers if not p[3] and o in p[5] and (not do_split or side(p, o) == "dn")]
            if sel:
                other_groups[(*key, o)].append(np.vstack([hold(*p[5][o][:2], grid) for p in sel]).mean(axis=0))
        if seeder_curve is None:
            seeder_curve = hold(r["seeder"][0], r["seeder"][1], grid)
    if n_skipped:
        print(f"note: {n_skipped} trial(s) skipped -- every peer was flagged slow")
    if split_chain and not n_split:
        print("note: --split-chain had no effect (no cascading trial with a slow peer)")

    ymax = float(seeder_curve.max())
    color_vals = sorted({k[0] for k in groups}, key=str)
    tint_vals = sorted({k[1] for k in groups}, key=lambda v: (isinstance(v, str), v))
    rank = {tv: i for i, tv in enumerate(tint_vals)}
    n_shades = len(tint_vals)

    def base(cv):
        return STRATEGY_COLORS.get(cv, plt.get_cmap("tab10")(color_vals.index(cv) % 10))

    if facet:
        fig, axes = plt.subplots(len(color_vals), 1, figsize=(8.6, 3.0 * len(color_vals) + 0.8),
                            sharex=True, sharey=True, squeeze=False,
                            gridspec_kw={"hspace": 0.28})
        axes = axes[:, 0]
        ax_for = {cv: axes[i] for i, cv in enumerate(color_vals)}

        # fig, axes = plt.subplots(1, len(color_vals), figsize=(5.0 * len(color_vals), 4.6),
        #                          sharex=True, sharey=True, squeeze=False,
        #                          gridspec_kw={"wspace": 0.22})
        # axes = axes[0]
        # ax_for = {cv: axes[i] for i, cv in enumerate(color_vals)}
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8.6, 5.4))
        else:
            fig = ax.figure
        axes = [ax]
        ax_for = {cv: ax for cv in color_vals}

    labels_for = defaultdict(list)
    for cv in color_vals:
        ax = ax_for[cv]
        mean_curves = {tv: np.vstack(groups[(cv, tv)]).mean(axis=0) for tv in tint_vals if (cv, tv) in groups}
        for tv, mc in mean_curves.items():
            if spread != "none":
                # repeats: spread of the per-trial mean peer over repeats (seed noise)
                # peers:   spread of individual healthy peers, pooled over repeats
                rows = np.vstack(groups[(cv, tv)] if spread == "repeats" else pool[(cv, tv)])
                if len(rows) > 1:
                    ax.fill_between(grid, rows.min(0), rows.max(0), color=shade(base(cv), rank[tv], n_shades),
                                    alpha=0.18, linewidth=0, zorder=1)
            ax.plot(grid, mc, color=shade(base(cv), rank[tv], n_shades), lw=1.8, zorder=3)
        for (c2, tv, o), curves in sorted(other_groups.items(), key=lambda kv: str(kv[0])):
            if c2 == cv:      # other swarms: same colour, dashed, no inline label
                ax.plot(grid, np.vstack(curves).mean(axis=0), color=shade(base(cv), rank[tv], n_shades),
                        lw=1.3, ls=(0, (5, 2.5)), zorder=2.5)
        for tv in tint_vals:
            if up_groups.get((cv, tv)):
                ax.plot(grid, np.vstack(up_groups[(cv, tv)]).mean(axis=0),
                        color=shade(base(cv), rank[tv], n_shades), lw=1.6, ls=(0, (1, 2)), zorder=5)

        pts = {tv: norm_pts(grid, mc, t_max, ymax) for tv, mc in mean_curves.items()}
        clusters = cluster_cells([v for v in tint_vals if v in mean_curves], pts, tol=0.03)
        cluster_pts = [pts[c[0]] for c in clusters]
        for k, cluster in enumerate(clusters):
            mc = mean_curves[cluster[0]]
            text = fmt_val(cluster[0]) if len(cluster) == 1 else f"{fmt_val(cluster[0])}\u2013{fmt_val(cluster[-1])}"
            done = mc >= ymax - 0.5
            if done.any():
                kind, order = "fin", (0, float(grid[int(np.argmax(done))]))
            else:
                kind, order = "end", (1, -float(mc[-1]))
            labels_for[id(ax)].append({"curve": mc, "text": text, "kind": kind, "order": order,
                                       "others": [p for j, p in enumerate(cluster_pts) if j != k],
                                       "color": mix(shade(base(cv), rank[cluster[0]], n_shades), (0, 0, 0), 0.30)})

    for ax in axes:
        ax.plot(grid, seeder_curve, color="black", lw=3.2, ls="-", zorder=2, solid_capstyle="butt")
        ax.set_xlim(0, 2500)
        # ax.set_xlim(0, t_max)
        ax.set_ylim(-0.04 * ymax, 1.06 * ymax)
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= ymax * 1.0001])
        ax.grid(alpha=0.25)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Number of Maps Held")
    if facet:
        for cv, ax in ax_for.items():
            ax.set_title(str(cv), fontsize=10)

    pol = [Patch(color=base(cv), label=str(cv)) for cv in color_vals]
    main_label = "mean healthy peer (excl. slow)"
    extra = []
    if n_split:
        main_label = "mean healthy peer; cascading: those\ndownstream of the slow node"
        extra = [Line2D([0], [0], color="#444", lw=1.0, ls=":",
                        label="cascading: healthy peers\nupstream of the slow node")]
    lines = [Line2D([0], [0], color="#444", lw=1.8, ls="-", label=main_label), *extra,
             Line2D([0], [0], color="black", lw=3.2, ls="-", label="seeder (published)"),
             Line2D([0], [0], color="none", label=f"line labels: {tint_by}"
                                                   f"\n(light = low, dark = high)")]
    host = axes[-1]
    if legend:
        leg1 = host.legend(handles=pol, title=color_by, fontsize=8, title_fontsize=8,
                           loc="upper center", bbox_to_anchor=(1.12, 1.0), frameon=False)
    keys = {'heterogeneity.slow_factor':'Radio Degradation Factor (%)',
            'agents.peers.0.count':'Number of Peers',
            'radio.fixed.d0':'Radio Strength (distance at which link rate is halved)'}
    ax.set_title(f'Sweep over {keys.get(tint_by)}')
    # host.add_artist(leg1)
    # host.legend(handles=lines, fontsize=8, loc="lower left", bbox_to_anchor=(1.12, 0.0), frameon=False)

    if labels_out is not None:
        labels_out.append((ax, labels_for[id(ax)], grid, ymax, {cv: base(cv) for cv in color_vals},
                           bool(other_groups), bool(up_groups)))
        return

    fig.subplots_adjust(left=0.07, right=0.80 if not facet else 0.84, bottom=0.12, top=0.92)
    for ax in axes:
        place_labels(fig, ax, labels_for[id(ax)], grid, ymax)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  ({len(raw) - n_skipped} trials, {len(groups)} policy x sweep cells, t_max={t_max:.0f}s)")


def picker_label(cv):
    return str(cv).replace("_", "-").capitalize()


def plot_progress_row(specs, out, figsize=(15.0, 4.4), letters=True, legend=True, **common):
    """One figure, one panel per sweep (1 x len(specs)), sharing the y-axis.
    Each spec is a dict of plot_progress kwargs for that panel (at least
    sweep_dir and tint_by; optionally tint_values, split_chain, ...);
    `common` kwargs (color_by, spread, split_chain, ...) apply to every panel
    unless a spec overrides them. `legend` adds one shared legend under the
    panels: title "Piece Picker", the seeder (black dashed) then each policy."""
    fig, axes = plt.subplots(1, len(specs), figsize=figsize, squeeze=False)
    axes = axes[0]
    pending = []
    for ax, spec in zip(axes, specs):
        kw = {"color_by": "agents.peers.0.strategy", **common, **spec}
        plot_progress(**kw, ax=ax, labels_out=pending, legend=False)
    same_y = len({round(p[3]) for p in pending}) == 1
    if same_y:      # one common scale: label the left panel only
        for ax in axes[1:]:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    if letters:
        for ax, letter in zip(axes, "abcdefgh"):
            ax.set_title(f"({letter}) {ax.get_title()}", fontsize=10)
    fig.tight_layout()                       # labels depend on the final axes size
    if not same_y:
        print("note: panels publish different numbers of maps "
              f"({sorted({round(p[3]) for p in pending})}); each keeps its own y-axis")
    for ax, labs, grid, ymax, *_ in pending:
        place_labels(fig, ax, labs, grid, ymax)
    if legend:
        colours = {}
        for p in pending:
            colours.update(p[4])
        multi = any(p[5] for p in pending)
        has_up = any(p[6] for p in pending)
        order = list(STRATEGY_COLORS)
        policies = sorted(colours, key=lambda p: (order.index(p) if p in order else len(order), str(p)))
        pick = [Line2D([0], [0], color="black", lw=3.2, label="Seeder"),
                *[Line2D([0], [0], color=colours[p], lw=3, label=picker_label(p)) for p in policies]]
        kw = dict(loc="upper left", frameon=False, fontsize=9, title_fontsize=10)
        legs = [fig.legend(handles=pick, title="Piece Picker", ncol=len(pick),
                           bbox_to_anchor=(0.0, 0.0), **kw)]
        if multi:   # line type = which swarm: only meaningful in the multi-swarm case
            swarm = [Line2D([0], [0], color="#444", lw=1.8, ls="-", label="Target"),
                     Line2D([0], [0], color="#444", lw=1.5, ls=(0, (5, 2.5)), label="Other")]
            if has_up:   # --split-chain: cascading peers upstream of the slow node, target swarm
                swarm.append(Line2D([0], [0], color="#444", lw=1.6, ls=(0, (1, 2)),
                                    label="Target, upstream of slow node"))
            legs.append(fig.legend(handles=swarm, title="Swarm", ncol=len(swarm),
                                   bbox_to_anchor=(0.0, 0.0), **kw))
        # centre the legend group under the panels: measure, then shift
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        widths = [inv.transform(l.get_window_extent())[1][0] - inv.transform(l.get_window_extent())[0][0]
                  for l in legs]
        gap = 0.04
        x = (1 - (sum(widths) + gap * (len(legs) - 1))) / 2
        for l, w in zip(legs, widths):
            l.set_bbox_to_anchor((x, 0.0), transform=fig.transFigure)
            x += w + gap
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  ({len(specs)} panels)")


def parse_val(s):
    s = s.strip()
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", help="Sweep output dir (has manifest.jsonl and trial_*/sim_log.json)")
    ap.add_argument("--color-by", default="agents.peers.0.strategy", help="Override key encoded as hue")
    ap.add_argument("--tint-by", required=True, help="Override key encoded as tint (the swept parameter)")
    ap.add_argument("--t-max", type=float, default=None, help="Right edge of the time axis (default: last event)")
    ap.add_argument("--facet", action="store_true", help="One panel per --color-by value")
    ap.add_argument("--split-chain", action="store_true",
                    help="Cascading trials with a slow peer: draw healthy peers downstream of the slow "
                         "node (solid) and upstream of it (dotted) separately instead of one blended mean")
    ap.add_argument("--spread", choices=["none", "repeats", "peers"], default="none",
                    help="Shade the spread behind each solid line: across repeats of the mean peer, "
                         "or across individual peers (default none)")
    ap.add_argument("--max-traces", type=int, default=6,
                    help="Max sweep values drawn per policy (default 6); longer sweeps are thinned evenly")
    ap.add_argument("--tint-values", default=None,
                    help="Comma-separated sweep values to draw, e.g. 100,200,300,500 (overrides --max-traces)")
    ap.add_argument("--out", default=None, help="Output path (.pdf or .png); default {sweep_dir}/progress.pdf")
    a = ap.parse_args()
    plot_progress(a.sweep_dir, a.color_by, a.tint_by, a.out or os.path.join(a.sweep_dir, "progress.pdf"),
                  t_max=a.t_max, facet=a.facet, max_traces=a.max_traces,
                  tint_values=[parse_val(s) for s in a.tint_values.split(",")] if a.tint_values else None,
                  split_chain=a.split_chain, spread=a.spread)