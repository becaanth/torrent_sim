import heapq
import random
import math
from collections import Counter
import json
import csv
import os
import itertools

import pdb

DIRECTIONS = {
    "N":  (0.0, 1.0),
    "NE": (0.7071, 0.7071),   # normalized -- must stay unit length
    "E":  (1.0, 0.0),
    "SE": (0.7071, -0.7071),
    "S":  (0.0, -1.0),
    "SW": (-0.7071, -0.7071),
    "W":  (-1.0, 0.0),
    "NW": (-0.7071, 0.7071),
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

    def sorted_participants(self):
        """Deterministic iteration order over participants"""
        return sorted(self.participants, key=lambda a: a.agent_id)

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

        # p(bad peer) following Fan et al.
        self.p_bad = 0.5

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
                "is_bad": a.is_bad,
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
        for peer in swarm.sorted_participants():
            if peer is downloader:
                continue
            if not any(t.torrent_id == t_id for t in peer.active_downloads):
                self.schedule(0.0, "PICK_PIECE", peer, data=t_id)

    def cancel_transfer(self, transfer, reason="churn"):
        """abort in-progress transfer (dont mark piece as completed)"""
        if transfer not in self.active_transfers:
            return
        t_id = transfer.torrent_id
        downloader = transfer.downloader
        uploader = transfer.uploader

        self.active_transfers.discard(transfer)
        downloader.active_downloads.discard(transfer)
        uploader.active_uploads.discard(transfer)
        downloader.downloading_pieces[t_id].discard(transfer.piece_id)

        self.log_event("TRANSFER_END", transfer_id=transfer.transfer_id, aborted=True, reason=reason)

        self.recalculate_bandwidth()
        # give downloader a chance to repick
        self.schedule(0.0, "PICK_PIECE", downloader, data=t_id)

    def all_work_done(self):
        """True once every swarm hsa finalized. used to stop churns self-reschduling"""
        if not all(s.is_finalized for s in self.swarms.values()):
            return False
        for agent in self.agents:
            for t_id in agent.strategies:
                if agent.finish_time.get(t_id) is None:
                    return False
        return True

    def start_churn(self, agent, p_bad, mean_down):
        """Kick off randomized on/off churn for a every peer
        p_bad is this peers long-run fraction of time spent unavailable
        """
        if p_bad <= 0.0:
            return  # no churn at all for this peer
        if p_bad >= 1.0:
            raise ValueError(f"churn p_bad must be < 1.0 (got {p_bad}) -- a peer that's "
                              f"always down can never contribute anything)")

        mean_up = mean_down * (1.0 - p_bad) / p_bad
        agent.churn_mean_up = mean_up
        agent.churn_mean_down = mean_down

        if random.random() < p_bad:
            agent.is_available = False
            self.log_event("CHURN_DEPART", agent_id=agent.agent_id)
            down = random.expovariate(1.0 / mean_down)
            self.schedule(down, "CHURN_REJOIN", agent, data=None)
        else:
            up = random.expovariate(1.0 / mean_up)
            self.schedule(up, "CHURN_DEPART", agent, data=None)


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
        for agent in swarm.sorted_participants():
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
            p for p in swarm.sorted_participants()
            if p is not seeder_agent and p.strategies[torrent_id]["strategy"] == "cascading"
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
        for agent in swarm.sorted_participants():
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

            elif event_type == "CHURN_DEPART":
                agent.is_available = False
                self.log_event("CHURN_DEPART", agent_id=agent.agent_id)
                # a dropped connection kills up/down
                for t in list(agent.active_uploads) + list(agent.active_downloads):
                    self.cancel_transfer(t, reason="churn")
                if not self.all_work_done():
                    down = random.expovariate(1.0 / agent.churn_mean_down)
                    self.schedule(down, "CHURN_REJOIN", agent, data=None)
            
            elif event_type == "CHURN_REJOIN":
                agent.is_available = True
                self.log_event("CHURN_REJOIN", agent_id=agent.agent_id)
                # wake up peers that this agent is a source
                for t_id in agent.strategies:
                    swarm = self.swarms[t_id]
                    for peer in swarm.sorted_participants():
                        if peer is not agent and not any(t.torrent_id == t_id for t in peer.active_downloads):
                            self.schedule(0.0, "PICK_PIECE", peer, data=t_id)
                if not self.all_work_done():
                    up = random.expovariate(1.0 / agent.churn_mean_up)
                    self.schedule(up, "CHURN_DEPART", agent, data=None)

            elif event_type == "MOVE_TICK":
                torrent_id, interval = data
                advanced = agent.try_walk_step(self, torrent_id)
                swarm = self.swarms[torrent_id]
                if (swarm.is_finalized and agent.walk_step >= swarm.published_pieces - 1):
                    if agent.walk_finish_time is None:
                        agent.walk_finish_time = self.time
                else:
                    # blocked, accrue wait time
                    if not advanced:
                        agent.wait_time += interval
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
        self.walk_finish_time = None  # sim time physical navigation actually completed (None until then)
        self.wait_time = 0.0
        
        # which swarms this agent seeds
        self.seeded_torrents = set()

        # churn/reliability
        self.is_bad = False
        self.is_available = True
        self.churn_mean_up = None
        self.churn_mean_down = None

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
        swarm.sorted_participants().add(self)

        if is_seeder:
            self.completed_pieces[t_id] = set(range(swarm.published_pieces))
            self.seeded_torrents.add(t_id)
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
        """try to advance on MOVE_TICK. Returns True if successful"""
        swarm = sim.swarms[torrent_id]
        next_step = self.walk_step + 1
        if next_step in self.completed_pieces[torrent_id]:
            self.walk_step = next_step
            self.position = swarm.piece_position(self.walk_step)
            sim.log_event("POSITION", agent_id=self.agent_id,
                x=self.position[0], y=self.position[1])
            return True
        return False

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
                if upstream and upstream.is_available and p in upstream.completed_pieces[torrent_id]:
                    r_i_sum += 1
            else:
                r_i_sum += sum(
                    1 for peer in swarm.sorted_participants() 
                    if peer is not self and peer.is_available and p in peer.completed_pieces[torrent_id]
                )
        return r_i_sum / swarm.published_pieces

    def get_metrics(self, sim, torrent_id, p_error=None):
        """Return Throughput [Mbps], Sequentiality [0..1], Robustness [0..1]"""
        # bad peer per Fan et al.
        if p_error is None:
            p_error = getattr(sim, "p_bad", 0.5)

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
        candidates = [p for p in swarm.sorted_participants() if p is not self and p.is_available]

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
            if upstream is not None and upstream.is_available:
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




def build_simulation_from_config(config):
    """Build a fully-wired, ready-to-run TorrentSim from a validated
    ScenarioConfig (see experiment_config.py). Reusable equivalent of the
    __main__ demo below, driven by config fields instead of hardcoded
    constants, so scenario/sweep YAMLs can build arbitrary trials without
    duplicating this setup logic per script.

    Returns (sim, all_agents, log_header) -- log_header is the pre-run
    replay-log snapshot, captured here since build_header() must run
    before any events are scheduled.
    """
    if config.simulation.seed is not None:
        random.seed(config.simulation.seed)

    sim = TorrentSim()
    sim.p_bad = config.churn.p_bad

    swarms = {}
    for name in config.topology.directions:
        swarms[name] = Swarm(
            torrent_id=name,
            direction=DIRECTIONS[name],
            initial_pieces=1,
            piece_size_mb=1,
            max_pieces=config.topology.horizon_for(name),
        )
        sim.add_swarm(swarms[name])

    all_agents = []
    next_agent_id = [0]

    def spawn_agent(radio_overrides=None, **kwargs):
        if config.radio.mode == "random":
            rr = config.radio.random_ranges
            profile = random_radio_profile(c_max_range=rr.c_max, d0_range=rr.d0, gamma_range=rr.gamma)
        else:
            f = config.radio.fixed
            profile = {"radio_c_max": f.c_max, "radio_d0": f.d0, "radio_gamma": f.gamma}
        for key, value in (radio_overrides or {}).items():
            profile[f"radio_{key}"] = value
        agent = Agent(agent_id=next_agent_id[0], **profile, **kwargs)
        next_agent_id[0] += 1
        all_agents.append(agent)
        return agent

    seeders = {}
    for seeder_spec in config.agents.seeders:
        name = seeder_spec.direction
        seeder = spawn_agent(position=(0.0, 0.0), radio_overrides=seeder_spec.radio_overrides)
        seeder.join_swarm(swarms[name], strategy="rarest_random", is_seeder=True, is_target=True)
        seeders[name] = seeder
        sim.start_seeder(seeder, name, interval=config.simulation.tick_interval)

    all_peers = []
    for group in config.agents.peers:
        target_cycle = itertools.cycle(group.target)  # round-robin, not random -- deterministic, even coverage
        for _ in range(group.count):
            target_dir = next(target_cycle)
            peer = spawn_agent(position=(0.0, 0.0), radio_overrides=group.radio_overrides)
            for name in config.topology.directions:
                peer.join_swarm(
                    swarms[name],
                    strategy=group.strategy,
                    hybrid_s=group.hybrid_s,
                    segment_k=group.segment_k,
                    is_target=(name == target_dir),
                )
            sim.start_walker(peer, target_dir, interval=config.simulation.move_interval)

            peer.is_bad = random.random() < config.churn.p_bad
            if peer.is_bad:
                sim.start_churn(peer, config.churn.p_bad, config.churn.mean_down)

            all_peers.append(peer)

    sim.agents = all_agents

    for name in config.topology.directions:
        sim.assign_cascade_chain(name, seeders[name])  # no-op if no cascading peers in this swarm

    log_header = sim.build_header()

    for agent in all_agents:
        for name in config.topology.directions:
            if name in agent.strategies:
                sim.schedule(0.0, "PICK_PIECE", agent, data=name)

    return sim, all_agents, log_header


def export_results(sim, all_agents, log_header, output_dir=".", write_log=True):
    """Write metrics.csv (always) and sim_log.json (if write_log) to
    output_dir. Same per-session TRS + throughput-ceiling logic as the
    __main__ demo, factored out so a runner script can call it once per
    trial without duplicating it. Returns the metrics row list (e.g. for
    a sweep runner to fold into a summary.csv across many trials)."""
    os.makedirs(output_dir, exist_ok=True)

    swarm_t_max = {}
    for t_id, swarm in sim.swarms.items():
        peers_in_swarm = [p for p in swarm.sorted_participants() if t_id not in p.seeded_torrents]
        seed_agents = [p for p in swarm.sorted_participants() if t_id in p.seeded_torrents]
        n_peers = len(peers_in_swarm)
        if n_peers == 0:
            swarm_t_max[t_id] = None
            continue
        u_s = seed_agents[0].upload_speed if seed_agents else 0.0
        u_p_avg = sum(p.upload_speed for p in peers_in_swarm) / n_peers
        swarm_t_max[t_id] = (u_s + n_peers * u_p_avg) / n_peers

    metrics_rows = []
    for agent in all_agents:
        for t_id in agent.strategies:
            is_seed = t_id in agent.seeded_torrents
            role = "SEED" if is_seed else ("TARGET" if t_id == agent.target_torrent_id else "relay")
            session_complete = agent.finish_time.get(t_id) is not None
            t, s, r = (None, None, None) if (is_seed or not session_complete) else agent.get_metrics(sim, t_id)

            metrics_rows.append({
                "agent_id": agent.agent_id,
                "torrent_id": t_id,
                "session_complete": session_complete,
                "target_torrent_id": agent.target_torrent_id,
                "role": role,
                "strategy": "(source)" if is_seed else agent.strategies[t_id]["strategy"],
                "is_bad": agent.is_bad,
                "throughput_mbps": t,
                "sequentiality": s,
                "robustness": r,
                "n_peers": None if is_seed else len(
                    [p for p in sim.swarms[t_id].participants if t_id not in p.seeded_torrents]),
                "t_max_mbps": None if is_seed else swarm_t_max[t_id],
                "start_time": agent.start_time[t_id],
                "finish_time": agent.finish_time[t_id],
                "walk_step": agent.walk_step if (not is_seed and t_id == agent.target_torrent_id) else None,
                "walk_finish_time": agent.walk_finish_time if (not is_seed and t_id == agent.target_torrent_id) else None,
                "total_pieces_held": (sum(len(agent.completed_pieces[tid]) for tid in agent.completed_pieces)
                                      if (not is_seed and t_id == agent.target_torrent_id) else None),
                "reachable_pieces_held": (sum(agent.contiguous_length(tid) for tid in agent.completed_pieces)
                                          if (not is_seed and t_id == agent.target_torrent_id) else None),
                "target_swarm_pieces_held": (len(agent.completed_pieces[agent.target_torrent_id])
                                              if (not is_seed and t_id == agent.target_torrent_id) else None),
                "target_swarm_reachable_pieces_held": (agent.contiguous_length(agent.target_torrent_id)
                                                        if (not is_seed and t_id == agent.target_torrent_id) else None),
                "wait_time": agent.wait_time/n_peers if (not is_seed and t_id == agent.target_torrent_id) else None,
            })


    fieldnames = ["agent_id", "torrent_id", "target_torrent_id", "role", "strategy", "is_bad", "session_complete",
                  "throughput_mbps", "sequentiality", "robustness", "n_peers", "t_max_mbps",
                  "start_time", "finish_time", "walk_step", "walk_finish_time", "total_pieces_held", "reachable_pieces_held", "target_swarm_pieces_held", "target_swarm_reachable_pieces_held", "wait_time"]
    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    if write_log:
        with open(os.path.join(output_dir, "sim_log.json"), "w") as f:
            json.dump({"header": log_header, "events": sim.log}, f)

    return metrics_rows



if __name__=="__main__":
  # Multi-directional mapping scenario demo. (For the general,
    # YAML-driven version of this same setup, see
    # build_simulation_from_config()/export_results() above -- this
    # block is a quick hardcoded smoke-test/demo entry point.)
    #
    # NOTE: this used to be accidentally duplicated twice inline in this
    # file (the whole setup+run+export, not just the export portion) --
    # that's exactly how the incomplete-session export fix ended up
    # applied inconsistently across copies, since any fix had to be
    # manually repeated in each copy and it was easy to miss one.
    # Collapsed back down to a single run that calls the shared
    # export_results() function.
    random.seed(42)  # reproducible heterogeneous radio spawns
    STRATEGY = "rarest_random"

    sim = TorrentSim()

    HORIZON_D = 100
    TICK_INTERVAL = 2.0
    MOVE_INTERVAL = 1.5   # seconds per physical unit-step, same for every agent
    MAX_SIM_TIME = 100000.0  # safety backstop -- see TorrentSim.run() docstring

    P_BAD = 0.2                # probability a given peer is "bad" (Fan et al.'s p)
    CHURN_MEAN_UP = 15.0       # mean seconds a bad peer stays available before dropping
    CHURN_MEAN_DOWN = 5.0      # mean seconds a bad peer stays gone before rejoining
    sim.p_bad = P_BAD          # read by get_metrics() for the reported robustness score

    # --- Build one swarm per direction ---
    swarms = {}
    for name, direction in DIRECTIONS.items():
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

        if use_random_radio:
            profile = random_radio_profile()
        else:
            profile = {
                "radio_c_max": 10.0,
                "radio_d0": 100.0,
                "radio_gamma": 2.0,
            }

        profile.update(radio_overrides)

        agent = Agent(agent_id=next_agent_id, **profile, **kwargs)
        next_agent_id += 1
        all_agents.append(agent)
        return agent

    # --- One seeder per direction, all starting at the shared root ---
    seeders = {}
    for name in DIRECTIONS:
        seeder = spawn_agent(use_random_radio=use_random_radio, position=(0.0, 0.0))
        seeder.join_swarm(swarms[name], strategy="rarest_random",
                           is_seeder=True, is_target=True)
        seeders[name] = seeder
        sim.start_seeder(seeder, name, interval=TICK_INTERVAL)

    # --- N peers per target path, each fully participating in every swarm ---
    # 4 peers per direction, across all 8 directions.
    peer_plan = []
    for d in DIRECTIONS:
        peer_plan += [(d, STRATEGY)] * 4

    peers = []
    for target_dir, strategy in peer_plan:
        peer = spawn_agent(use_random_radio=use_random_radio, position=(0.0, 0.0))
        for name in DIRECTIONS:
            # Every peer joins every swarm (full participation), but only
            # its target direction is marked is_target=True so that
            # progress there is what actually moves it through space.
            peer.join_swarm(
                swarms[name],
                strategy=strategy,
                is_target=(name == target_dir),
            )
        sim.start_walker(peer, target_dir, interval=MOVE_INTERVAL)
        peer.is_bad = random.random() < P_BAD
        if peer.is_bad:
            sim.start_churn(peer, P_BAD, CHURN_MEAN_DOWN)

        peers.append(peer)

    sim.agents = all_agents

    # Lock in each swarm's cascade chain now, once, before anything runs
    # -- a single random shuffle per swarm, never touched again.
    for name in DIRECTIONS:
        sim.assign_cascade_chain(name, seeders[name])

    log_header = sim.build_header()

    # Kick off piece-picking for every agent across every swarm it joined.
    for agent in all_agents:
        for name in DIRECTIONS:
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

    # Single call to the shared export function -- writes
    # {STRATEGY}/metrics.csv and {STRATEGY}/sim_log.json.
    metrics_rows = export_results(sim, all_agents, log_header, output_dir=STRATEGY, write_log=True)

    print("\n" + "=" * 100)
    print("PER-SESSION TRS METRICS")
    print("Each agent gets one row per swarm it participated in (target path")
    print("plus every swarm it relayed for), not just its own target.")
    print("Incomplete sessions (no finish_time) are included too -- see")
    print("session_complete in the CSV.")
    print("=" * 100)
    print(f"{'Agent ID':<10} | {'Swarm':<6} | {'Role':<8} | {'Strategy':<15} | {'Throughput':<12} | {'Sequentiality':<14} | {'Robustness'}")
    print("-" * 100)
    for row in metrics_rows:
        if row["role"] == "SEED":
            print(f"Agent {row['agent_id']:<4} | {row['torrent_id']:<6} | {'SEED':<8} | "
                  f"{'(source)':<15} | {'n/a':<12} | {'n/a':<14} | n/a")
        elif not row["session_complete"]:
            # throughput_mbps/sequentiality/robustness are blank (None)
            # for incomplete sessions -- formatting them as floats would
            # crash, so this branch handles that case explicitly.
            print(f"Agent {row['agent_id']:<4} | {row['torrent_id']:<6} | {row['role']:<8} | {row['strategy']:<15} | "
                  f"{'incomplete':<12} | {'--':<14} | --")
        else:
            print(f"Agent {row['agent_id']:<4} | {row['torrent_id']:<6} | {row['role']:<8} | {row['strategy']:<15} | "
                  f"{row['throughput_mbps']:6.2f} Mbps  | {row['sequentiality']:6.4f}       | {row['robustness']:6.4f}")

    print(f"\nOutput written to {STRATEGY}/metrics.csv and {STRATEGY}/sim_log.json")