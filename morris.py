"""Morris (elementary effects) sensitivity analysis for torrent_sim.py.

Pipeline: sample -> run each trial through the simulator -> reduce each
trial's per-session metrics.csv rows to one scalar per output metric ->
compute elementary effects (mu, mu*, sigma) via SALib -> plot mu* vs
sigma per output.

Uses SALib for sampling/analysis (the standard, validated Morris
implementation) rather than hand-rolled trajectory generation.

IMPORTANT: Morris needs continuous/ordered factors -- it works by taking
small steps along each parameter and measuring the resulting change in
output. `strategy` is categorical and doesn't fit that model, so it is
NOT a Morris parameter here; it's fixed as part of the base scenario. To
compare sensitivity across strategies, run this whole pipeline once per
base scenario (one per fixed strategy), not as a single combined study.

Study YAML:
    base: ../scenarios/base_scenario.yaml
    trajectories: 20        # SALib's N -- more trajectories = tighter
                             # confidence intervals, more trials to run
    levels: 4                # SALib's num_levels (grid resolution per factor)
    role_filter: TARGET      # which row role to reduce each trial down from
    write_logs: false
    parameters:
      churn.p_bad: {bounds: [0.0, 0.5]}
      simulation.move_interval: {bounds: [0.3, 8.0]}
      radio.fixed.d0: {bounds: [3.0, 100.0]}
      radio.fixed.gamma: {bounds: [1.0, 3.5]}
      agents.peers.0.count: {bounds: [4, 40], integer: true}
    outputs:
      - finish_time
      - map_possession_fraction           # normalized [0,1]: avg per-agent own-path pieces held / horizon
      - reachable_map_possession_fraction # normalized [0,1]: avg per-agent own-path contiguous pieces / horizon
      - wait_time                         # SUMMED across peers: total fleet-wide time spent stalled

Total trials = trajectories * (num_parameters + 1). Keep the base
scenario cheap (small horizon/peer count) -- Morris needs many trials by
design, this is a screening tool, not where you'd run your final
hero-scenario numbers.

Usage:
    python3 morris_analysis.py morris_studies/sequentiality_study.yaml
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_config import ScenarioConfig, _set_dotted
from run_simulation import run_one, run_trials_parallel

# Metrics where "None" (an incomplete session) gets substituted with the
# scenario's max_sim_time rather than dropped -- a right-censoring
# convention: "never finished within budget" is itself informative and
# excluding it would bias exactly the parameter regions where things fail.
CENSORED_AT_MAX_TIME = {"finish_time", "walk_finish_time"}

# Metrics reported as a SUM across TARGET peers rather than a mean --
# e.g. wait_time, where the question is "how much total fleet-time was
# spent stalled," which should scale with fleet size, not be averaged
# away by it. Everything else defaults to a per-agent mean.
SUMMED_METRICS = {"wait_time"}


class MorrisParameter(BaseModel):
    bounds: tuple[float, float]
    integer: bool = False

    @model_validator(mode="after")
    def _check_bounds(self):
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(f"bounds must be (low, high) with low < high, got {self.bounds}")
        return self


class MorrisConfig(BaseModel):
    base: str
    parameters: dict[str, MorrisParameter]
    trajectories: int = 20
    levels: int = 4
    role_filter: str = "TARGET"
    outputs: list[str] = Field(default_factory=lambda: [
        "finish_time", "map_possession_fraction", "reachable_map_possession_fraction", "wait_time"])
    write_logs: bool = False

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


def generate_samples(morris_config):
    problem = {
        "num_vars": len(morris_config.parameters),
        "names": list(morris_config.parameters.keys()),
        "bounds": [list(p.bounds) for p in morris_config.parameters.values()],
    }
    X = morris_sample.sample(problem, N=morris_config.trajectories, num_levels=morris_config.levels)
    return problem, X


def build_trial_config(base_dict, problem, row, morris_config):
    merged = copy.deepcopy(base_dict)
    resolved = {}
    for name, value in zip(problem["names"], row):
        if morris_config.parameters[name].integer:
            value = int(round(value))
        else:
            value = float(value)
        _set_dotted(merged, name, value)
        resolved[name] = value
    return ScenarioConfig.model_validate(merged), resolved


# Derived metrics: normalized into a [0,1] fraction of the relevant
# swarm's own horizon, not a raw count. A raw target_swarm_pieces_held
# count is meaningless on its own (12 out of how many?), and more
# importantly, comparing raw counts across trials that vary peer count
# invites exactly the confound "more agents -> more total capacity" even
# though reduce_trial already averages per-agent, not sums -- what's
# missing is knowing what fraction of the achievable maximum that
# average represents. Maps each derived name to its underlying raw
# column.
DERIVED_METRICS = {
    "map_possession_fraction": "target_swarm_pieces_held",
    "reachable_map_possession_fraction": "target_swarm_reachable_pieces_held",
}


def reduce_trial(rows, outputs, role_filter, max_sim_time, topology_config=None):
    """Collapse one trial's full metrics.csv rows down to one scalar per
    requested output metric: summed across every row matching
    role_filter for outputs in SUMMED_METRICS (e.g. wait_time), averaged
    otherwise (there may be several matching rows, e.g. multiple TARGET
    peers).

    Outputs in DERIVED_METRICS are computed by first normalizing EACH
    row's raw value against that row's own swarm horizon (via
    topology_config.horizon_for(row['torrent_id'])) before aggregating --
    not by aggregating raw counts and normalizing once at the end -- so
    this stays correct even if different TARGET rows belong to swarms
    with different horizon_overrides.
    """
    target_rows = [r for r in rows if r["role"] == role_filter]
    result = {}
    for out in outputs:
        base_metric = DERIVED_METRICS.get(out, out)
        vals = []
        for r in target_rows:
            v = r.get(base_metric)
            if v is None and base_metric in CENSORED_AT_MAX_TIME:
                v = max_sim_time
            if v is None:
                continue
            v = float(v)
            if out in DERIVED_METRICS:
                if topology_config is None:
                    raise ValueError(f"'{out}' requires topology_config to normalize against horizon")
                horizon = topology_config.horizon_for(r["torrent_id"])
                v = v / horizon if horizon else float("nan")
            vals.append(v)
        if not vals:
            result[out] = float("nan")
        elif out in SUMMED_METRICS:
            result[out] = sum(vals)
        else:
            result[out] = sum(vals) / len(vals)
    return result


def run_trials_for_base(base_dict, problem, X, morris_config, out_dir, label="", n_jobs=1):
    """Run every row of a Morris sample matrix X against one fully-
    resolved base_dict, returning {output_name: [value_per_trial]}
    aligned to X's row order (required by SALib.analyze.morris). Shared
    by both run_morris_study (single base) and
    run_morris_compare_strategies (same X, multiple strategy-overridden
    bases) so there's exactly one place this logic lives.

    Trials run via run_trials_parallel (processes, not threads -- see
    its docstring). Each trial's ScenarioConfig is built here in the
    parent up front (cheap: just dict overrides + pydantic validation),
    so reduce_trial can use config.topology/max_sim_time after results
    come back, without needing to carry the config object back across
    the process boundary alongside the (possibly large) metrics rows.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_trials = len(X)
    prefix = f"{label}: " if label else ""

    configs, resolved_by_index, specs = {}, {}, []
    for i, row in enumerate(X):
        config, resolved = build_trial_config(base_dict, problem, row, morris_config)
        configs[i] = config
        resolved_by_index[i] = resolved
        trial_dir = os.path.join(out_dir, f"trial_{i:05d}")
        specs.append((i, config, trial_dir, morris_config.write_logs))

    Y = {out: [None] * n_trials for out in morris_config.outputs}
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    completed = 0
    with open(manifest_path, "w") as manifest_f:
        for trial_index, rows in run_trials_parallel(specs, n_jobs=n_jobs):
            config = configs[trial_index]
            reduced = reduce_trial(rows, morris_config.outputs, morris_config.role_filter,
                                    config.simulation.max_sim_time, topology_config=config.topology)
            for out in morris_config.outputs:
                Y[out][trial_index] = reduced[out]

            manifest_f.write(json.dumps({"trial_index": trial_index,
                                          "params": resolved_by_index[trial_index],
                                          "outputs": reduced}) + "\n")
            completed += 1
            if completed % max(1, n_trials // 20) == 0 or completed == n_trials:
                print(f"  {prefix}{completed}/{n_trials} trials complete")

    np.save(os.path.join(out_dir, "X.npy"), X)
    with open(os.path.join(out_dir, "problem.json"), "w") as f:
        json.dump(problem, f)
    for out in morris_config.outputs:
        np.save(os.path.join(out_dir, f"Y_{out}.npy"), np.array(Y[out]))

    return Y


def run_morris_study(config_path, out_dir=None, n_jobs=1):
    morris_config = MorrisConfig.from_yaml(config_path)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    base_path = morris_config.base if os.path.isabs(morris_config.base) else os.path.join(base_dir, morris_config.base)
    with open(base_path) as f:
        base_dict = yaml.safe_load(f)
    base_name = base_dict["scenario"]["name"]

    out_dir = out_dir or os.path.join("results", f"morris_{base_name}")

    problem, X = generate_samples(morris_config)
    print(f"Morris study on '{base_name}': {len(problem['names'])} parameters, "
          f"{morris_config.trajectories} trajectories -> {len(X)} trials, jobs={n_jobs}")
    print(f"Parameters: {problem['names']}")

    Y = run_trials_for_base(base_dict, problem, X, morris_config, out_dir, n_jobs=n_jobs)
    return problem, X, Y, out_dir, morris_config


def run_morris_compare_strategies(config_path, strategies, out_dir=None, n_jobs=1):
    """Run the same Morris study (same sampled parameter trajectories,
    for a fair paired comparison) once per strategy, by overriding the
    base scenario's peer-group strategy field each time -- mirrors
    run_simulation.py's --compare-strategies. Requires the base scenario
    to have exactly one peer group (the "picker under test"), same
    constraint as the single-strategy tool. n_jobs parallelizes the
    TRIALS within each strategy's run_trials_for_base call, not the
    strategies themselves -- each strategy still runs one after another,
    but its own trials run in parallel."""
    morris_config = MorrisConfig.from_yaml(config_path)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    base_path = morris_config.base if os.path.isabs(morris_config.base) else os.path.join(base_dir, morris_config.base)
    with open(base_path) as f:
        base_dict = yaml.safe_load(f)

    n_groups = len(base_dict.get("agents", {}).get("peers", []))
    if n_groups != 1:
        print(f"--compare-strategies requires a base scenario with exactly one peer group "
              f"(the picker under test); '{morris_config.base}' has {n_groups}.", file=sys.stderr)
        sys.exit(1)

    base_name = base_dict["scenario"]["name"]
    root = out_dir or os.path.join("results", f"morris_{base_name}")
    os.makedirs(root, exist_ok=True)

    problem, X = generate_samples(morris_config)  # sampled once, reused for every strategy
    print(f"Morris comparison on '{base_name}': {len(problem['names'])} parameters, "
          f"{morris_config.trajectories} trajectories -> {len(X)} trials PER strategy "
          f"({len(X) * len(strategies)} total), same sample reused across strategies, jobs={n_jobs}")

    all_results = {}
    for strategy in strategies:
        strat_dict = copy.deepcopy(base_dict)
        strat_dict["agents"]["peers"][0]["strategy"] = strategy
        strat_dir = os.path.join(root, strategy)

        Y = run_trials_for_base(strat_dict, problem, X, morris_config, strat_dir, label=strategy, n_jobs=n_jobs)
        results = analyze_morris(problem, X, Y, strat_dir, num_levels=morris_config.levels)
        plot_morris(results, strat_dir)
        all_results[strategy] = results

    combined_rows = []
    for strategy, results in all_results.items():
        for out, Si in results.items():
            for name, mu, mu_star, mu_star_conf, sigma in zip(
                    Si["names"], Si["mu"], Si["mu_star"], Si["mu_star_conf"], Si["sigma"]):
                combined_rows.append({"strategy": strategy, "output": out, "parameter": name,
                                       "mu": mu, "mu_star": mu_star, "mu_star_conf": mu_star_conf, "sigma": sigma})
    combined_csv_path = os.path.join(root, "morris_comparison.csv")
    pd.DataFrame(combined_rows).to_csv(combined_csv_path, index=False)
    print(f"Wrote {combined_csv_path}")

    plot_morris_comparison(all_results, morris_config.outputs, root)
    return all_results, root


def analyze_morris(problem, X, Y, out_dir, num_levels):
    results = {}
    for out, y_list in Y.items():
        y_arr = np.array(y_list, dtype=float)
        n_nan = int(np.isnan(y_arr).sum())
        if n_nan:
            print(f"[warn] {out}: {n_nan}/{len(y_arr)} trial(s) produced no data at all "
                  f"(no matching rows, not just incomplete) -- filling with the column mean "
                  f"as a fallback so analysis can proceed. Check the manifest if this is large.")
            y_arr = np.where(np.isnan(y_arr), np.nanmean(y_arr), y_arr)

        Si = morris_analyze.analyze(problem, X, y_arr, print_to_console=False, num_levels=num_levels)
        results[out] = Si

        df = pd.DataFrame({
            "parameter": Si["names"],
            "mu": Si["mu"],
            "mu_star": Si["mu_star"],
            "mu_star_conf": Si["mu_star_conf"],
            "sigma": Si["sigma"],
        }).sort_values("mu_star", ascending=False)
        csv_path = os.path.join(out_dir, f"morris_{out}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

    return results


def plot_morris(results, out_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), squeeze=False)
    axes = axes[0]

    for ax, (out, Si) in zip(axes, results.items()):
        ax.scatter(Si["mu_star"], Si["sigma"], s=60, zorder=3)
        for name, mx, sy in zip(Si["names"], Si["mu_star"], Si["sigma"]):
            ax.annotate(name, (mx, sy), fontsize=8, xytext=(5, 5), textcoords="offset points")
        # sigma/mu* = 1 reference line: above it, effects vary a lot
        # trajectory-to-trajectory (nonlinear/interacting); below it,
        # the parameter's effect is fairly consistent wherever you are
        # in the space.
        xmax = max(Si["mu_star"]) * 1.15 if len(Si["mu_star"]) else 1.0
        ax.plot([0, xmax], [0, xmax], linestyle="--", color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel("mu* (overall influence)")
        ax.set_ylabel("sigma (nonlinearity / interaction)")
        ax.set_title(out, fontsize=11)
        ax.grid(alpha=0.3)

    fig.suptitle("Morris elementary effects", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "morris_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_morris_comparison(all_results, outputs, out_dir):
    """Same mu*/sigma scatter as plot_morris, but with every strategy
    overlaid (colored) on the same axes per output metric, so you can
    see directly whether a given parameter matters differently for
    different pickers."""
    strategies = list(all_results.keys())
    cmap = plt.get_cmap("tab10")
    colors = {s: cmap(i % 10) for i, s in enumerate(strategies)}

    fig, axes = plt.subplots(1, len(outputs), figsize=(6.5 * len(outputs), 5.5), squeeze=False)
    axes = axes[0]

    for ax, out in zip(axes, outputs):
        all_mu_star = []
        for strategy in strategies:
            Si = all_results[strategy][out]
            ax.scatter(Si["mu_star"], Si["sigma"], color=colors[strategy], s=60, zorder=3, label=strategy)
            for name, mx, sy in zip(Si["names"], Si["mu_star"], Si["sigma"]):
                ax.annotate(name, (mx, sy), fontsize=7, xytext=(4, 4),
                            textcoords="offset points", color=colors[strategy])
            all_mu_star.extend(Si["mu_star"])

        xmax = max(all_mu_star) * 1.15 if all_mu_star else 1.0
        ax.plot([0, xmax], [0, xmax], linestyle="--", color="gray", alpha=0.4, linewidth=1)
        ax.set_xlabel("mu* (overall influence)")
        ax.set_ylabel("sigma (nonlinearity / interaction)")
        ax.set_title(out, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="Strategy")

    fig.suptitle("Morris elementary effects -- strategy comparison", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "morris_comparison_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a Morris study YAML")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--compare-strategies", nargs="+", default=None, metavar="STRATEGY",
                         help="Run the same Morris study once per strategy listed here (e.g. "
                              "--compare-strategies rarest_random sequential cascading), reusing "
                              "the same sampled parameter trajectories across all of them for a "
                              "fair comparison. Requires the base scenario to have exactly one "
                              "peer group.")
    parser.add_argument("--jobs", type=int, default=1,
                         help="Number of trials to run in parallel, via separate processes. "
                              "With --compare-strategies, applies within each strategy's own "
                              "trial batch (strategies still run one after another).")
    args = parser.parse_args()

    if args.compare_strategies:
        run_morris_compare_strategies(args.config, args.compare_strategies, out_dir=args.out_dir, n_jobs=args.jobs)
    else:
        problem, X, Y, out_dir, morris_config = run_morris_study(args.config, out_dir=args.out_dir, n_jobs=args.jobs)
        results = analyze_morris(problem, X, Y, out_dir, num_levels=morris_config.levels)
        plot_morris(results, out_dir)
        print(f"\nDone. See {out_dir}/morris_*.csv and {out_dir}/morris_plot.png")