"""Run one scenario or an entire sweep of trials from a YAML config.

Usage:
    python3 run_simulation.py scenarios/cascading_negative_churn.yaml
    python3 run_simulation.py sweeps/cascading_p_bad_sweep.yaml

A scenario file (has a top-level `scenario:` key) runs once, writing
metrics.csv + sim_log.json to results/{scenario.name}/.

A sweep file (has top-level `base:` + `trials:` keys) expands into many
trials, each writing to results/{scenario.name}/trial_{i:05d}/. By
default only metrics.csv is written per trial (sim_log.json is opt-in via
the sweep's `write_logs: true`, since it's expensive at trial counts in
the thousands). Two extra files are written at the sweep root:
  - manifest.jsonl: one line per trial, recording its exact resolved
    overrides -- essential once seeds/p_bad/etc. are randomized, since
    you need to know exactly what produced each trial's numbers.
  - summary.csv: every trial's metrics.csv rows unioned into one table,
    with trial_index and the varied parameter values as extra columns --
    the actual analysis artifact at scale, rather than opening N files.
"""
import argparse
import copy
import csv
import json
import os
import sys

import yaml

from experiment_config import ScenarioConfig, SweepConfig
import torrent_sim


def run_one(config, output_dir, write_log=True):
    sim, all_agents, log_header = torrent_sim.build_simulation_from_config(config)
    sim.run(max_time=config.simulation.max_sim_time)
    return torrent_sim.export_results(sim, all_agents, log_header, output_dir=output_dir, write_log=write_log)


def _run_one_trial(spec):
    """Top-level, picklable worker for ProcessPoolExecutor. spec is
    (trial_index, config, output_dir, write_log); returns
    (trial_index, metrics_rows). Must stay top-level (not a closure/
    lambda) -- multiprocessing pickles the callable by reference."""
    trial_index, config, output_dir, write_log = spec
    rows = run_one(config, output_dir, write_log=write_log)
    return trial_index, rows


def run_trials_parallel(specs, n_jobs=1):
    """Run a batch of independent trials, each spec a (trial_index,
    config, output_dir, write_log) tuple, yielding (trial_index, rows)
    as each completes (NOT necessarily in trial_index order -- callers
    that need order, e.g. Morris's Y arrays, should index into a
    pre-sized list/array by trial_index rather than relying on yield
    order).

    n_jobs<=1 runs sequentially in-process (no pool overhead). n_jobs>1
    uses a ProcessPoolExecutor, deliberately NOT threads: every trial
    seeds and draws from the global `random` module
    (torrent_sim.build_simulation_from_config calls random.seed(...)
    directly), and threads share that global state across trials --
    concurrent threads would race on the same RNG stream and silently
    corrupt each other's draws. Separate processes each get their own
    fresh interpreter and `random` module state, so per-trial seeding
    (each trial's config already carries its own resolved seed from
    resolve_sweep/Morris sampling) stays correct with no extra work.
    """
    if n_jobs <= 1:
        for spec in specs:
            yield _run_one_trial(spec)
        return

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_run_one_trial, spec) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def run_compare_strategies(path, strategies):
    """Load one base scenario and run it once per strategy in
    `strategies`, overriding only the peer group's strategy field --
    topology, churn, radio, everything else stays identical, so the
    comparison isolates just the picker. Requires the base scenario to
    have exactly one peer group (the "picker under test"); a scenario
    with several groups already mixing strategies -- like
    mixed_strategy_comparison.yaml -- doesn't have a single strategy
    field to swap, so it's out of scope for this mode.

    Writes each variant to results/{scenario.name}/{strategy}/ (full
    metrics.csv + sim_log.json, single-run semantics), and also drops a
    flat results/{scenario.name}/{strategy}.csv copy at the scenario
    root -- gem_metrics.py/metrics_plots.py already default to reading
    exactly rarest_random.csv/sequential.csv/cascading.csv by name, so
    this plugs straight into the existing plotting scripts with no
    renaming needed.
    """
    import shutil

    with open(path) as f:
        base_dict = yaml.safe_load(f)

    n_groups = len(base_dict.get("agents", {}).get("peers", []))
    if n_groups != 1:
        print(f"--compare-strategies requires a base scenario with exactly one peer group "
              f"(the picker under test); '{path}' has {n_groups}.", file=sys.stderr)
        sys.exit(1)

    scenario_name = base_dict["scenario"]["name"]
    root = os.path.join("results", scenario_name)
    os.makedirs(root, exist_ok=True)

    flat_paths = {}
    for strategy in strategies:
        variant_dict = copy.deepcopy(base_dict)
        variant_dict["agents"]["peers"][0]["strategy"] = strategy
        config = ScenarioConfig.model_validate(variant_dict)

        variant_dir = os.path.join(root, strategy)
        print(f"Running '{scenario_name}' with strategy={strategy} -> {variant_dir}")
        run_one(config, variant_dir, write_log=True)

        flat_path = os.path.join(root, f"{strategy}.csv")
        shutil.copy(os.path.join(variant_dir, "metrics.csv"), flat_path)
        flat_paths[strategy] = flat_path

    print(f"\nDone. {len(strategies)} strategy variants of '{scenario_name}' written under {root}/")
    print("Ready for the comparison plots:")
    print(f"  python3 gem_metrics.py --metrics {' '.join(flat_paths.values())}")
    print(f"  python3 metrics_plots.py --metrics {' '.join(flat_paths.values())}")


def run_scenario(path):
    config = ScenarioConfig.from_yaml(path)
    output_dir = os.path.join("results", config.scenario.name)
    print(f"Running scenario '{config.scenario.name}' ({config.scenario.hypothesis or 'unlabeled'} "
          f"example for {config.scenario.target_strategy or '?'}) -> {output_dir}")
    # write_log is always True for a single scenario run (as opposed to a
    # sweep, where it's opt-in per-sweep) -- this is what makes it
    # possible to render a video of what happened afterward.
    rows = run_one(config, output_dir, write_log=True)
    log_path = os.path.join(output_dir, "sim_log.json")
    video_path = os.path.join(output_dir, "render.mp4")
    print(f"Done. {len(rows)} session rows written to {output_dir}/metrics.csv")
    print(f"Replay log written to {log_path}")
    print(f"To render a video of this run:\n"
          f"  python3 render_sim.py {log_path} --out {video_path}")


def run_sweep(path, n_jobs=1):
    from experiment_config import resolve_sweep

    sweep = SweepConfig.from_yaml(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    trials = resolve_sweep(sweep, base_dir=base_dir)
    n_trials = len(trials)

    scenario_name = ScenarioConfig.from_yaml(
        sweep.base if os.path.isabs(sweep.base) else os.path.join(base_dir, sweep.base)
    ).scenario.name
    sweep_root = os.path.join("results", scenario_name)
    os.makedirs(sweep_root, exist_ok=True)

    manifest_path = os.path.join(sweep_root, "manifest.jsonl")
    summary_rows = []

    print(f"Running sweep over '{scenario_name}': {n_trials} trials "
          f"({len(sweep.grid)} grid axis(es) x {sweep.repeats} repeat(s)), "
          f"varying {list(sweep.vary.keys())}, logs {'ON' if sweep.write_logs else 'off'}, "
          f"jobs={n_jobs}")

    # overrides are looked up by trial_index at write time, since
    # run_trials_parallel only carries (config, output_dir, write_log)
    # across the process boundary -- the human-readable overrides dict
    # stays in the parent process.
    overrides_by_index = {trial_index: overrides for trial_index, _, overrides in trials}
    specs = [(trial_index, config, os.path.join(sweep_root, f"trial_{trial_index:05d}"), sweep.write_logs)
             for trial_index, config, _ in trials]

    completed = 0
    with open(manifest_path, "w") as manifest_f:
        for trial_index, rows in run_trials_parallel(specs, n_jobs=n_jobs):
            overrides = overrides_by_index[trial_index]
            manifest_f.write(json.dumps({"trial_index": trial_index, "overrides": overrides}) + "\n")
            for row in rows:
                row = dict(row)
                row["trial_index"] = trial_index
                row.update(overrides)
                summary_rows.append(row)

            completed += 1
            if completed % max(1, n_trials // 20) == 0 or completed == n_trials:
                print(f"  {completed}/{n_trials} trials complete")

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        summary_path = os.path.join(sweep_root, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Done. {len(summary_rows)} rows across {n_trials} trials -> {summary_path}")
    else:
        print("Done, but no session rows were produced across any trial -- check the scenario config.")


def is_sweep_file(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return "base" in raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a scenario YAML or a sweep YAML")
    parser.add_argument("--compare-strategies", nargs="+", default=None,
                         metavar="STRATEGY",
                         help="Run the given scenario once per strategy listed here (e.g. "
                              "--compare-strategies rarest_random sequential cascading), "
                              "overriding only the picker -- everything else in the scenario "
                              "stays identical. Requires the scenario to have exactly one "
                              "peer group.")
    parser.add_argument("--jobs", type=int, default=1,
                         help="Number of trials to run in parallel (sweep files only -- a "
                              "single scenario run or --compare-strategies is just 1-3 runs, "
                              "not worth pooling). Uses separate processes, not threads.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"No such file: {args.config}", file=sys.stderr)
        sys.exit(1)

    if args.compare_strategies:
        run_compare_strategies(args.config, args.compare_strategies)
    elif is_sweep_file(args.config):
        run_sweep(args.config, n_jobs=args.jobs)
    else:
        run_scenario(args.config)