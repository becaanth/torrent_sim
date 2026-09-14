"""Pydantic schema + YAML loading for torrent_sim.py experiment configs.

Two file types:
  - A scenario file fully specifies one simulation (see ScenarioConfig).
  - A sweep file wraps a base scenario and describes how to vary it
    across many trials (see SweepConfig) -- for the "thousands of
    randomized trials" case. Sweep files never duplicate the base
    scenario's fields; they only list dotted-path overrides and how to
    sample them per trial.
"""
from typing import Literal, Optional
import copy

import yaml
from pydantic import BaseModel, Field, model_validator

KNOWN_STRATEGIES = {"rarest_random", "sequential", "cascading", "hybrid", "segment_random"}

# Kept in sync with torrent_sim.DIRECTIONS by name (not imported directly,
# so this module has no hard dependency on torrent_sim.py -- validation
# here only needs to know which names are legal, not their vectors).
ALL_DIRECTIONS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


# --------------------------------------------------------------------------
# Scenario schema
# --------------------------------------------------------------------------

class ScenarioMeta(BaseModel):
    name: str
    description: str = ""
    target_strategy: Optional[str] = None
    hypothesis: Optional[Literal["positive", "negative"]] = None

    @model_validator(mode="after")
    def _check_strategy(self):
        if self.target_strategy is not None and self.target_strategy not in KNOWN_STRATEGIES:
            raise ValueError(f"target_strategy '{self.target_strategy}' not in {sorted(KNOWN_STRATEGIES)}")
        return self


class SimulationConfig(BaseModel):
    seed: Optional[int] = None
    max_sim_time: float = 500.0
    tick_interval: float = 2.0
    move_interval: float = 1.5


class TopologyConfig(BaseModel):
    directions: list[str] = Field(default_factory=lambda: ["N", "E", "S", "W"])
    horizon: int = 8
    horizon_overrides: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_directions(self):
        bad = set(self.directions) - ALL_DIRECTIONS
        if bad:
            raise ValueError(f"unknown direction(s) {sorted(bad)}; valid: {sorted(ALL_DIRECTIONS)}")
        bad_overrides = set(self.horizon_overrides) - set(self.directions)
        if bad_overrides:
            raise ValueError(f"horizon_overrides references direction(s) not in directions: {sorted(bad_overrides)}")
        return self

    def horizon_for(self, direction):
        return self.horizon_overrides.get(direction, self.horizon)


class ChurnConfig(BaseModel):
    p_bad: float = 0.0
    mean_up: float = 15.0
    mean_down: float = 5.0

    @model_validator(mode="after")
    def _check_p(self):
        if not (0.0 <= self.p_bad <= 1.0):
            raise ValueError(f"churn.p_bad must be within [0, 1], got {self.p_bad}")
        return self


class RadioProfile(BaseModel):
    c_max: float = 10.0
    d0: float = 10.0
    gamma: float = 2.0


class RadioRanges(BaseModel):
    c_max: tuple[float, float] = (6.0, 14.0)
    d0: tuple[float, float] = (5.0, 15.0)
    gamma: tuple[float, float] = (1.5, 3.0)


class RadioConfig(BaseModel):
    mode: Literal["fixed", "random"] = "fixed"
    fixed: RadioProfile = Field(default_factory=RadioProfile)
    random_ranges: RadioRanges = Field(default_factory=RadioRanges)


class SeederSpec(BaseModel):
    direction: str
    radio_overrides: dict[str, float] = Field(default_factory=dict)


class PeerGroupSpec(BaseModel):
    count: int = 1
    target: list[str]
    strategy: str
    hybrid_s: float = 0.5
    segment_k: int = 5
    radio_overrides: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_group(self):
        if self.strategy not in KNOWN_STRATEGIES:
            raise ValueError(f"strategy '{self.strategy}' not in {sorted(KNOWN_STRATEGIES)}")
        if self.count <= 0:
            raise ValueError("agents.peers[].count must be positive")
        if not self.target:
            raise ValueError("agents.peers[].target must list at least one direction")
        return self


class AgentsConfig(BaseModel):
    seeders: list[SeederSpec]
    peers: list[PeerGroupSpec]


class ScenarioConfig(BaseModel):
    scenario: ScenarioMeta
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    churn: ChurnConfig = Field(default_factory=ChurnConfig)
    radio: RadioConfig = Field(default_factory=RadioConfig)
    agents: AgentsConfig

    @model_validator(mode="after")
    def _check_targets_in_directions(self):
        dirs = set(self.topology.directions)
        for seeder in self.agents.seeders:
            if seeder.direction not in dirs:
                raise ValueError(f"seeder direction '{seeder.direction}' not in topology.directions {sorted(dirs)}")
        seeded = {s.direction for s in self.agents.seeders}
        for group in self.agents.peers:
            bad = set(group.target) - dirs
            if bad:
                raise ValueError(f"peer target(s) {sorted(bad)} not in topology.directions {sorted(dirs)}")
        missing_seeds = dirs - seeded
        if missing_seeds:
            raise ValueError(f"direction(s) {sorted(missing_seeds)} have no seeder -- every "
                              f"topology.direction needs exactly one entry in agents.seeders")
        return self

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# --------------------------------------------------------------------------
# Sweep schema
# --------------------------------------------------------------------------

class VarySpec(BaseModel):
    """How to sample one dotted-path config field across trials."""
    distribution: Literal["uniform", "choice", "trial_index"]
    low: Optional[float] = None
    high: Optional[float] = None
    values: Optional[list] = None

    @model_validator(mode="after")
    def _check_params(self):
        if self.distribution == "uniform" and (self.low is None or self.high is None):
            raise ValueError("distribution: uniform requires low and high")
        if self.distribution == "choice" and not self.values:
            raise ValueError("distribution: choice requires a non-empty values list")
        return self

    def sample(self, rng, trial_index, base_seed):
        if self.distribution == "uniform":
            return rng.uniform(self.low, self.high)
        if self.distribution == "choice":
            return rng.choice(self.values)
        if self.distribution == "trial_index":
            return (base_seed or 0) + trial_index
        raise ValueError(f"unhandled distribution '{self.distribution}'")  # pragma: no cover


class SweepConfig(BaseModel):
    base: str
    trials: int
    vary: dict[str, VarySpec]
    write_logs: bool = False  # opt-in per sweep; off by default given trial counts

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


def _set_dotted(d, dotted_key, value):
    parts = dotted_key.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def resolve_sweep(sweep, base_dir="."):
    """Expand a SweepConfig into a list of (trial_index, ScenarioConfig,
    overrides_dict) tuples, one per trial. Each trial's overrides are
    sampled from a dedicated random.Random seeded off the base scenario's
    seed (or 0), so resolution itself is reproducible independent of
    whatever global `random` state torrent_sim.py's own simulation RNG is
    in when this runs."""
    import os
    import random as _random

    base_path = sweep.base if os.path.isabs(sweep.base) else os.path.join(base_dir, sweep.base)
    with open(base_path) as f:
        base_dict = yaml.safe_load(f)
    base_seed = (base_dict.get("simulation") or {}).get("seed") or 0

    rng = _random.Random(base_seed)
    resolved = []
    for trial_index in range(sweep.trials):
        overrides = {key: spec.sample(rng, trial_index, base_seed) for key, spec in sweep.vary.items()}
        merged = copy.deepcopy(base_dict)
        for dotted_key, value in overrides.items():
            _set_dotted(merged, dotted_key, value)
        scenario = ScenarioConfig.model_validate(merged)
        resolved.append((trial_index, scenario, overrides))
    return resolved