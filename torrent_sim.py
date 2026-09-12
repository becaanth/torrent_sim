import heapq
import random
import math
from collections import Counter
import json

import pdb

CARDINAL_DIRECTIONS = {
    "N": (0.0, 1.0),
    "E": (1.0, 0.0),
    "S": (0.0, -1.0),
    "W": (-1.0, 0.0),
}

def random_radio_profile(rng=random, c_max_range=(6.0, 14.0),
                          d0_range=(5.0, 15.0), gamma_range=(1.5, 3.0)):
    """sample a random radio"""
    return {
        "radio_c_max": rng.uniform(*c_max_range),
        "radio_d0": rng.uniform(*d0_range),
        "radio_gamma": rng.uniform(*gamma_range),
    }

class Swarm:
    """Represents a single torrent session"""
    def __init__(self, torrent_id, direction, initial_pieces=1, piece_size_mb=1, max_pieces=None):
        self.torrent_id = torrent_id
        self.direction = direction
        self.published_pieces = initial_pieces
        self.piece_size_mb = piece_size_mb
        self.max_pieces = max_pieces
        self.is_finalized = False # flag for closing the stream
        self.participants = set()

    def piece_position(self, piece_id):
        """give a spatial position for the piece"""
        return (self.direction[0] * piece_id, self.direction[1] * piece_id)

class Transfer:
    def __init__(self, transfer_id, torrent_id, downloader, uploader, piece_id, total_size_mb):
        self.transfer_id = transfer_id
        self.torrent_id = torrent_id
        self.downloader = downloader
        self.uploader = uploader
        self.piece_id = piece_id
        self.remaining_mb = total_size_mb
        self.current_rate_mbps = 0.0
        self.last_update_time = 0.0
        self.scheduled_finish_time = float('inf')
        
class TorrentSim:
    """Operates as the global discrete event engine and network manager"""
    def __init__(self, c_max=10.0, d0=10.0, gamma=2.0):
        self.swarms = {} # torrent_id -> swarm
        self.time = 0.0
        self.counter = 0  # Unique tie-breaker sequence number
        self.transfer_counter = 0
        self.event_queue = []  # (time, event_type, peer, data)
        self.active_transfers = set()
        self.agents = []

        # spatial channel params (nominally can set heterogeneity per-agent)
        self.c_max = c_max # max radio throughput
        self.d0 = d0
        self.gamma = gamma

        # replay log
        self.log = []

    # world-level setup
    def add_swarm(self, swarm):
        self.swarms[swarm.torrent_id] = swarm

    def log_event(self, event_type, **fields):
        self.log.append({"t": self.time, "type":event_type, **fields})

    def build_header(self):
        """Snapshot everything the replay/render script needs that never
        changes once the run starts: swarm geometry, the agent roster,
        and each agent's pre-run piece ownership"""
        swarms = {
            tid: {"direction": list(s.direction), "horizon_D": s.max_pieces}
            for tid, s in self.swarms.items()
        }
        agents = []
        initial_pieces = {}
        for a in self.agents:
            agents.append({
                "id": a.agent_id,
                "seeded_torrents": sorted(a.seeded_torrents),
                "target_torrent_id": a.target_torrent_id,
                "strategies": {tid: info["strategy"] for tid, info in a.strategies.items()},
                "start_position": list(a.position),
            })
            initial_pieces[a.agent_id] = {
                tid: sorted(pieces) for tid, pieces in a.completed_pieces.items()
            }
        return {"swarms": swarms, "agents": agents, "initial_pieces": initial_pieces}

    def compute_link_capacity(self, agent_a, agent_b):
        """radio capacity as a function of distance"""
        dx = agent_a.position[0] - agent_b.position[0]
        dy = agent_a.position[1] - agent_b.position[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # path-loss attenuation (w/ heterogenous radios)
        c_max = min(agent_a.radio_c_max, agent_b.radio_c_max)
        d0 = (agent_a.radio_d0 + agent_b.radio_d0) / 2.0
        gamma = (agent_a.radio_gamma + agent_b.radio_gamma) / 2.0

        capacity = c_max / (1.0 + (dist/d0)**gamma)
        return dist, capacity

    def schedule(self, delay, event_type, agent, data=None):
        self.counter += 1
        heapq.heappush(
            self.event_queue, 
            (self.time + delay, self.counter, event_type, agent, data)
        )

    # handling network model
    def _update_transfer_progress(self):
        """Advance bytes transferred for active streams up to this sim time"""
        for t in self.active_transfers:
            elapsed = self.time - t.last_update_time
            if elapsed > 0 and t.current_rate_mbps > 0:
                mb_transferred = (t.current_rate_mbps * elapsed) / 8.0
                t.remaining_mb = max(0.0, t.remaining_mb - mb_transferred)
            t.last_update_time = self.time

    # per-channel chunk capacity following Fan et al.
    def recalculate_bandwidth(self):
        """
        Dynamically divide uploader/downloader capcaity among active streams per Fan et al. per-chunk capacity formulation
        """
        self._update_transfer_progress()

        for t in self.active_transfers:
            # uploader capacity split uniformly across uploads
            uploader_share = t.uploader.upload_speed / len(t.uploader.active_uploads)
            downloader_share = t.downloader.download_speed / len(t.downloader.active_downloads)

            _, link_cap = self.compute_link_capacity(t.uploader, t.downloader)

            # bottleneck rate for this transfer
            t.current_rate_mbps = min(uploader_share, downloader_share, link_cap)

            if t.current_rate_mbps > 0:
                remaining_bits = t.remaining_mb * 8.0
                time_to_complete = remaining_bits / t.current_rate_mbps
                t.scheduled_finish_time = self.time + time_to_complete

                # push updated finish time to event queue
                self.schedule(time_to_complete, "PIECE_COMPLETE", t.downloader, data=t)

    def start_transfer(self, downloader, uploader, torrent_id, piece_id):
        self.transfer_counter += 1
        swarm = self.swarms[torrent_id]
        transfer = Transfer(
            self.transfer_counter, torrent_id, 
            downloader, uploader, 
            piece_id, swarm.piece_size_mb
        )

        downloader.downloading_pieces[torrent_id].add(piece_id)
        downloader.active_downloads.add(transfer)
        uploader.active_uploads.add(transfer)
        self.active_transfers.add(transfer)

        self.log_event("TRANSFER_START", transfer_id=transfer.transfer_id,
                        torrent_id=torrent_id, piece_id=piece_id,
                        downloader_id=downloader.agent_id, uploader_id=uploader.agent_id)

        self.recalculate_bandwidth()

    def finish_transfer(self, transfer):
        t_id = transfer.torrent_id
        swarm = self.swarms[t_id]
        downloader = transfer.downloader

        self.active_transfers.remove(transfer)
        downloader.active_downloads.remove(transfer)
        transfer.uploader.active_uploads.remove(transfer)
        self.log_event("TRANSFER_END", transfer_id=transfer.transfer_id)

        downloader.downloading_pieces[t_id].remove(transfer.piece_id)
        downloader.completed_pieces[t_id].add(transfer.piece_id)
        self.log_event("PIECE_OWNED", agent_id=downloader.agent_id,
                        torrent_id=t_id, piece_id=transfer.piece_id)
        
        # record metrics
        downloader.record_useful_chunks(self,t_id)
        self.recalculate_bandwidth()

        self.schedule(0.0, "PICK_PIECE", downloader, data=t_id)
        # wake up any idle neighbours in this swarm (Fixes cascading pipeline stall)
        for neighbours in swarm.participants:
            if peer is downloader:
                continue
            if not any(t.torrent_id == t_id for t in neighbours.active_downloads):
                self.schedule(0.0, "PICK_PIECE", neighbours, data=t_id)

    def append_pieces(self, torrent_id, count, publisher_agent):
        """Append new pieces to the end of the seed (Append-only Mutable Torrent)"""
        swarm = self.swarms[torrent_id]
        new_start = swarm.published_pieces
        swarm.published_pieces += count
        new_pieces = set(range(new_start, swarm.published_pieces))

        publisher_agent.completed_pieces[torrent_id].update(new_pieces)
        for p in sorted(new_pieces):
            self.log_event("PIECE_OWNED", agent_id=publisher_agent.agent_id,
                torrent_id=torrent_id, piece_id=p)


        # advance seeder to its newest map
        publisher_agent.position = swarm.piece_position(swarm.published_pieces - 1)
        self.log_event("POSITION", agent_id=publisher_agent.agent_id,
                        x=publisher_agent.position[0], y=publisher_agent.position[1])
        self.log_event("APPEND", torrent_id=torrent_id, horizon=swarm.published_pieces)

        print(f"[Time {self.time:6.2f}s] APPEND: Swarm '{torrent_id}' +{count} pieces added. "
              f"New stream horizon: [0 .. {swarm.published_pieces - 1}]")
        
        # wake up idle leechers
        for agent in swarm.participants:
            if agent != publisher_agent: 
                self.schedule(0.0, "PICK_PIECE", agent, data=torrent_id)

    def start_seeder(self, seeder_agent, torrent_id, interval, pieces_per_tick=1):
        """recurring publisher timer for a seeder"""
        self.schedule(interval, "SEEDER_TICK", seeder_agent,
                      data=(torrent_id, interval, pieces_per_tick))

    def start_walker(self, agent, torrent_id, interval):
        """Leechers walking clock (enabling repeats)"""
        self.schedule(interval, "MOVE_TICK", agent, data=(torrent_id, interval))

    def assign_cascade_chain(self, torrent_id, seeder_agent, rng=random):
        """IMO faithful cascading implementation - random connectivity chains at session start"""
        swarm = self.swarms[torrent_id]
        cascading_peers = [
            p for p in swarm.participants
            if p is not seeder and p.strategies[torrent_id]["strategy"] == "cascading"
        ]
        rng.shuffle(cascading_peers)

        chain = [seeder_agent] + cascading_peers
        for upstream, downstream in zip(chain, chain[1:]):
            downstream.upstream_peers[torrent_id] = upstream

    def finalize_stream(self, torrent_id):
        """Mark the stream as complete"""
        swarm = self.swarms[torrent_id]
        swarm.is_finalized = True
        self.log_event("FINALIZE", torrent_id=torrent_id)
        print(f"[Time {self.time:6.2f}s] FINALIZE: Swarm '{torrent_id}': Closed at {swarm.published_pieces} total pieces.")

        # Trigger idle leechers to evaluate total completion
        for agent in swarm.participants:
            if not any(t.torrent_id == torrent_id for t in agent.active_downloads):
                self.schedule(0.0, "PICK_PIECE", agent, data=torrent_id)
        
    def run(self, max_time=None):
        while self.event_queue:
            if max_time is not None and self.event_queue[0][0] > max_time:
                print(f"[SAFETY STOP] max_time={max_time}s reached with "
                    f"{len(self.event_queue)} events still queued -- halting early.")
                break

            self.time, _, event_type, agent, data = heapq.heappop(self.event_queue)

            if event_type == "PICK_PIECE":
                torrent_id = data
                agent.pick_next_piece(self, torrent_id)

            elif event_type == "APPEND_PIECES":
                torrent_id, count = data
                self.append_pieces(torrent_id, count, agent)

            elif event_type == "MOVE_TICK":
                torrent_id, interval = data
                agent.try_walk_step(self, torrent_id)
                swarm = self.swarms[torrent_id]
                if not (swarm.is_finalized and agent.walk_step >= swarm.published_pieces - 1):
                    self.schedule(interval, "MOVE_TICK", agent, data=(torrent_id, interval))

            elif event_type == "SEEDER_TICK":
                torrent_id, interval, pieces_per_tick = data
                swarm = self.swarms[torrent_id]
                if swarm.is_finalized:
                    continue

                pieces_to_add = pieces_per_tick
                if swarm.max_pieces is not None:
                    remaining_horizon = swarm.max_pieces - swarm.published_pieces
                    pieces_to_add = max(0, min(pieces_per_tick, remaining_horizon))

                if pieces_to_add > 0:
                    self.append_pieces(torrent_id, pieces_to_add, agent)

                if swarm.max_pieces is not None and swarm.published_pieces >= swarm.max_pieces:
                    self.finalize_stream(torrent_id)
                else: # keep ticking
                    self.schedule(interval, "SEEDER_TICK", agent,
                                  data=(torrent_id, interval, pieces_per_tick))

            elif event_type == "FINALIZE_STREAM":
                torrent_id = data
                self.finalize_stream(torrent_id)

            elif event_type == "PIECE_COMPLETE":
                transfer = data
                # filter stale events
                if abs(self.time - transfer.scheduled_finish_time) > 1e-7:
                    continue

                if transfer in self.active_transfers:
                    t_id = transfer.torrent_id
                    swarm = self.swarms[t_id]
                    self.finish_transfer(transfer)

                    if len(agent.completed_pieces[t_id]) == swarm.published_pieces and swarm.is_finalized:
                        if agent.finish_time[t_id] is None:
                            agent.finish_time[t_id] = self.time
                            print(f"[Time {self.time:6.2f}s] Agent {agent.agent_id:2d} "
                                  f"COMPLETED Swarm '{t_id}' ({swarm.published_pieces}/{swarm.published_pieces})")
                    else:
                        self.schedule(0.0, "PICK_PIECE", agent, data=t_id)

class Agent:
    """all agents in a swarm"""
    def __init__(self, agent_id, position=(0.0, 0.0), download_speed_mbps=10, upload_speed_mbps=10,
            radio_c_max=10.0, radio_d0=10.0, radio_gamma=2.0            
        ):
        self.agent_id = agent_id
        self.position = position # (x,y)
        self.download_speed = download_speed_mbps
        self.upload_speed = upload_speed_mbps

        # per-agent radio hardware
        self.radio_c_max = radio_c_max
        self.radio_d0 = radio_d0
        self.radio_gamma = radio_gamma

        # repeat we are executing
        self.target_torrent_id = None
        self.walk_step = 0

        # which swarms this agent seeds
        self.seeded_torrents = set()

        # continuous time tracking sets
        self.active_uploads = set()
        self.active_downloads = set()

        # state dicts keyed by torrent_id
        self.completed_pieces = {}
        self.downloading_pieces = {}
        self.neighbours = {}
        self.strategies = {}
        self.upstream_peers = {}

        # metrics per swarm
        self.start_time = {}
        self.finish_time = {}
        self.u_x_history = {} # history of useful chunks
        self.r_bar_snapshots = {} # history of robustness

    def join_swarm(self, swarm, strategy="rarest_random", hybrid_s=0.5, segment_k=5, upstream_peer=None, is_seeder=False, is_target=False):
        t_id = swarm.torrent_id
        swarm.participants.add(self)

        if is_seeder:
            self.completed_pieces[t_id] = set(range(swarm.published_pieces))
        else:
            self.completed_pieces[t_id] = {0} # hold root node for free

        self.downloading_pieces[t_id] = set()
        self.neighbours[t_id] = []
        self.strategies[t_id] = {
            "strategy": strategy,
            "hybrid_s": hybrid_s,
            "segment_k": segment_k
        }
        self.upstream_peers[t_id] = upstream_peer

        # record who we're tracking
        if is_target:
            self.target_torrent_id = t_id

        self.start_time[t_id] = None
        self.finish_time[t_id] = None
        self.u_x_history[t_id] = []
        self.r_bar_snapshots[t_id] = []

    def connect(self, other_agent, torrent_id):
        if other_agent not in self.neighbours[torrent_id]:
            self.neighbours[torrent_id].append(other_agent)
        if self not in other_agent.neighbours[torrent_id]:
            other_agent.neighbours[torrent_id].append(self)

    def contiguous_length(self, torrent_id):
        """how many contiguous maps do we hold"""
        completed = self.completed_pieces[torrent_id]
        length = 0
        while length in completed:
            length += 1
        return length

    def distance_to(self, other):
        """tie-breaker for cascading"""
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def try_walk_step(self, sim, torrent_id):
        """try to advance on MOVE_TICK"""
        swarm = sim.swarms[torrent_id]
        next_step = self.walk_step + 1
        if next_step in self.completed_pieces[torrent_id]:
            self.walk_step = next_step
            self.position = swarm.piece_position(self.walk_step)
            sim.log_event("POSITION", agent_id=self.agent_id,
                x=self.position[0], y=self.position[1])

    def record_useful_chunks(self, sim, torrent_id):
        """U(x) per Eq. 10 in Fan et al."""
        completed = self.completed_pieces[torrent_id]
        x = len(completed)
        contiguous_len = self.contiguous_length(torrent_id)

        u_x = contiguous_len / x if x > 0 else 0.0
        self.u_x_history[torrent_id].append(u_x)

        r_bar = self.compute_current_r_bar(sim,torrent_id)
        self.r_bar_snapshots[torrent_id].append(r_bar)

    def compute_current_r_bar(self, sim, torrent_id):
        """r_bar per Eq. 8 in Fan et al."""
        swarm = sim.swarms[torrent_id]
        if swarm.published_pieces == 0:
            return 0.0
        r_i_sum = 0
        strat = self.strategies[torrent_id]["strategy"]
        for p in range(swarm.published_pieces):
            if strat == "cascading":
                upstream = self.upstream_peers.get(torrent_id)
                if upstream and p in upstream.completed_pieces[torrent_id]:
                    r_i_sum += 1
            else:
                r_i_sum += sum(
                    1 for peer in swarm.participants 
                    if p in peer.completed_pieces[torrent_id]
                )
        return r_i_sum / swarm.published_pieces

    def get_metrics(self, sim, torrent_id, p_error=0.5):
        """Return Throughput [Mbps], Sequentiality [0..1], Robustness [0..1]"""
        swarm = sim.swarms[torrent_id]
        start = self.start_time.get(torrent_id)
        finish = self.finish_time.get(torrent_id)
        if start is None or finish is None:
            return 0.0,0.0,0.0

        # throughput
        duration = finish - start
        file_size_bits = len(self.completed_pieces[torrent_id]) * swarm.piece_size_mb * 8.0
        throughput = file_size_bits / duration if duration > 0 else 0.0

        # sequentiality (eq. 10)
        hist = self.u_x_history.get(torrent_id, [])
        sequentiality = sum(hist) / swarm.published_pieces if hist else 0.0

        # robustness (eq. 8)
        r_snaps = self.r_bar_snapshots.get(torrent_id, [])
        avg_r_bar = sum(r_snaps) / len(r_snaps) if r_snaps else 0.0
        robustness = 1.0 - (p_error ** avg_r_bar)

        return throughput, sequentiality, robustness

    def pick_next_piece(self, sim, torrent_id):
        if torrent_id not in self.completed_pieces:
            return

        # Guard: If agent is already actively downloading in this swarm, wait for that download to finish
        if any(t.torrent_id == torrent_id for t in self.active_downloads):
            return
        
        if self.start_time[torrent_id] is None:
            self.start_time[torrent_id] = sim.time

        swarm = sim.swarms[torrent_id]
        missing = set(range(swarm.published_pieces)) - self.completed_pieces[torrent_id] - self.downloading_pieces[torrent_id]
        if not missing:
            # Handle stream completion if finalized while idle
            if len(self.completed_pieces[torrent_id]) == swarm.published_pieces and swarm.is_finalized:
                if self.finish_time[torrent_id] is None:
                    self.finish_time[torrent_id] = sim.time
                    print(f"[Time {sim.time:6.2f}s] Agent {self.agent_id:2d}"
                        f"COMPLETED full Swarm '{torrent_id}' ({swarm.published_pieces}/{swarm.published_pieces})")
            return

        strat_info = self.strategies[torrent_id]
        strategy = strat_info["strategy"]
        hybrid_s = strat_info["hybrid_s"]
        segment_k = strat_info["segment_k"]

        chosen_piece = None
        target_agent = None

        # bitmap discovery is global (like zenoh gossip), only transfer will throttle
        candidates = [p for p in swarm.participants if p is not self]

        # RAREST-RANDOM
        if strategy == "rarest_random":
            available_pieces = Counter()
            peer_map = {} # piece_id -> list of peers who have it
            for n in candidates:
                for p in n.completed_pieces[torrent_id]:
                    if p in missing:
                        available_pieces[p] += 1
                        peer_map.setdefault(p, []).append(n)

            if available_pieces:
                min_freq = min(available_pieces.values())
                rarest_candidates = [p for p, count in available_pieces.items() if count == min_freq]
                chosen_piece = random.choice(rarest_candidates)
                target_agent = random.choice(peer_map[chosen_piece])

        # SEQUENTIAL
        elif strategy == "sequential":
            available_pieces = set()
            peer_map = {}
            for n in candidates:
                for p in n.completed_pieces[torrent_id]:
                    if p in missing:
                        available_pieces.add(p)
                        peer_map.setdefault(p, []).append(n)

            if available_pieces:
                chosen_piece = min(available_pieces)
                target_agent = random.choice(peer_map[chosen_piece])

        # CASCADING
        elif strategy == "cascading":
            # check if current bound target peer still has missing pieces to offer
            upstream = self.upstream_peers.get(torrent_id)
            if upstream is not None:
                available = upstream.completed_pieces[torrent_id] & missing
                if available:
                    chosen_piece = min(available)
                    target_agent = upstream

        # HYBRID (sequential w/ prob s, random w/ prob 1-s)
        elif strategy == 'hybrid':
            available_pieces = Counter()
            peer_map = {}
            for n in candidates:
                for p in n.completed_pieces[torrent_id]:
                    if p in missing:
                        available_pieces[p] += 1
                        peer_map.setdefault(p, []).append(n)

            if available_pieces:
                if random.random() < hybrid_s:
                    chosen_piece = min(available_pieces.keys())
                else:
                    min_freq = min(available_pieces.values())
                    rarest = [p for p, count in available_pieces.items() if count == min_freq]
                    chosen_piece = random.choice(rarest)

                target_agent = random.choice(peer_map[chosen_piece])

        # SEGMENT-RANDOM (bucket sequential, intra-bucket random)
        elif strategy == 'segment_random':
            available_pieces = Counter()
            peer_map = {}
            for n in candidates:
                for p in n.completed_pieces[torrent_id]:
                    if p in missing:
                        available_pieces[p] += 1
                        peer_map.setdefault(p, []).append(n)

            if available_pieces:
                buckets = {}
                for p in available_pieces:
                    b_idx = p // segment_k
                    buckets.setdefault(b_idx, []).append(p)

                earliest_bucket_idx = min(buckets.keys())
                bucket_candidates = buckets[earliest_bucket_idx]

                candidate_freqs = {p: available_pieces[p] for p in bucket_candidates}
                min_freq = min(candidate_freqs.values())
                rarest_in_bucket = [p for p, count in candidate_freqs.items() if count == min_freq]

                chosen_piece = random.choice(rarest_in_bucket)
                target_agent = random.choice(peer_map[chosen_piece])
       
        # dispatch task if a valid piece and peer target were selected
        if chosen_piece is not None and target_agent is not None:
            sim.start_transfer(
                downloader=self,
                uploader=target_agent,
                torrent_id=torrent_id,
                piece_id=chosen_piece
            )

if __name__=="__main__":
    # NEW DEMO: multi-directional mapping scenario.
    #
    # Four seeders map outward from a shared root (0,0), one per cardinal
    # direction, each its own swarm/torrent. Several peer robots each
    # target one direction (their physical path) while still fully
    # participating (downloading + relaying) in every other swarm. Radio
    # hardware is heterogeneous per-agent via random_radio_profile(), and
    # each swarm has a fixed mapping horizon D so the run terminates
    # predictably.
    random.seed(42)  # reproducible heterogeneous radio spawns

    sim = TorrentSim()

    HORIZON_D = 50
    TICK_INTERVAL = 2.0
    MOVE_INTERVAL = 1.5   # seconds per physical unit-step, same for every agent
    MAX_SIM_TIME = 5000.0  # safety backstop -- see TorrentSim.run() docstring

    # --- Build one swarm per cardinal direction ---
    swarms = {}
    for name, direction in CARDINAL_DIRECTIONS.items():
        swarms[name] = Swarm(
            torrent_id=name,
            direction=direction,
            initial_pieces=1,       # just the shared root to start
            piece_size_mb=1,
            max_pieces=HORIZON_D,
        )
        sim.add_swarm(swarms[name])

    all_agents = []
    next_agent_id = 0
    use_random_radio = False

    def spawn_agent(use_random_radio=True, **kwargs):
        """Helper: assign the next agent_id and either a randomized radio profile
        or default Agent radio parameters unless overridden."""
        global next_agent_id
        radio_overrides = kwargs.pop("radio_overrides", {})
        
        profile = {}
        if use_random_radio:
            profile = random_radio_profile()
            
        profile.update(radio_overrides)
        
        agent = Agent(agent_id=next_agent_id, **profile, **kwargs)
        next_agent_id += 1
        all_agents.append(agent)
        return agent

    # --- One seeder per direction, all starting at the shared root ---
    seeders = {}
    for name in CARDINAL_DIRECTIONS:
        seeder = spawn_agent(use_random_radio=use_random_radio, position=(0.0, 0.0))
        seeder.join_swarm(swarms[name], strategy="rarest_random",
                           is_seeder=True, is_target=True)
        seeders[name] = seeder
        sim.start_seeder(seeder, name, interval=TICK_INTERVAL)

    # --- N peers per target path, each fully participating in every swarm ---
    # (direction, strategy) pairs — feel free to vary these to compare
    # piece-picking strategies head-to-head on the same map.
    peer_plan = [
        ("N", "rarest_random"),
        ("N", "rarest_random"),
        ("E", "rarest_random"),
        ("S", "rarest_random"),
        ("S", "rarest_random"),
        ("W", "rarest_random"),
    ]

    peers = []
    for target_dir, strategy in peer_plan:
        peer = spawn_agent(use_random_radio=use_random_radio, position=(0.0, 0.0))
        for name in CARDINAL_DIRECTIONS:
            # Every peer joins every swarm (full participation), but only
            # its target direction is marked is_target=True so that
            # progress there is what actually moves it through space.
            peer.join_swarm(
                swarms[name],
                strategy=strategy,
                is_target=(name == target_dir),
            )
        sim.start_walker(peer, target_dir, interval=MOVE_INTERVAL)
        peers.append(peer)

    sim.agents = all_agents

    # NEW: lock in each swarm's cascade chain now, once, before anything
    # runs -- a single random shuffle per swarm, never touched again.
    for name in CARDINAL_DIRECTIONS:
        sim.assign_cascade_chain(name, seeders[name])

    log_header = sim.build_header()

    # Kick off piece-picking for every agent across every swarm it joined.
    for agent in all_agents:
        for name in CARDINAL_DIRECTIONS:
            if name in agent.strategies:
                sim.schedule(0.0, "PICK_PIECE", agent, data=name)

    print("=" * 90)
    print("MULTI-DIRECTIONAL MAPPING SIMULATION")
    print(f"Horizon D={HORIZON_D} pieces/direction, tick interval={TICK_INTERVAL}s")
    print("=" * 90)
    for agent in all_agents:
        print(f"Agent {agent.agent_id:2d} | radio c_max={agent.radio_c_max:5.2f} "
              f"d0={agent.radio_d0:5.2f} gamma={agent.radio_gamma:4.2f} "
              f"| target={agent.target_torrent_id}")
    print("-" * 90)

    sim.run(max_time=MAX_SIM_TIME)

    print("\n" + "=" * 100)
    print("PER-SESSION TRS METRICS")
    print("Each agent gets one row per swarm it participated in (target path")
    print("plus every swarm it relayed for), not just its own target.")
    print("=" * 100)
    print(f"{'Agent ID':<10} | {'Swarm':<6} | {'Role':<8} | {'Strategy':<15} | {'Throughput':<12} | {'Sequentiality':<14} | {'Robustness'}")
    print("-" * 100)
    for agent in all_agents:
        for t_id in agent.strategies:
            if agent.start_time.get(t_id) is None or agent.finish_time.get(t_id) is None:
                continue  # this session never completed (or never started)
            t, s, r = agent.get_metrics(sim, t_id)
            role = "TARGET" if t_id == agent.target_torrent_id else "relay"
            strategy = agent.strategies[t_id]["strategy"]
            print(f"Agent {agent.agent_id:<4} | {t_id:<6} | {role:<8} | {strategy:<15} | "
                  f"{t:6.2f} Mbps  | {s:6.4f}       | {r:6.4f}")

    # Write the replay log for the separate render_sim.py script.
    log_path = "sim_log.json"
    with open(log_path, "w") as f:
        json.dump({"header": log_header, "events": sim.log}, f)
    print(f"\nReplay log written to {log_path} ({len(sim.log)} events).")