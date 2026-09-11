import heapq
import random
import math
from collections import Counter

import pdb

class Swarm:
    """Represents a single torrent session"""
    def __init__(self, torrent_id, initial_pieces=10, piece_size_mb=1):
        self.torrent_id = torrent_id
        self.published_pieces = initial_pieces
        self.piece_size_mb = piece_size_mb
        self.is_finalized = False # flag for closing the stream
        self.participants = set()

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

        # spatial channel params
        self.c_max = c_max # max radio throughput
        self.d0 = d0
        self.gamma = gamma

    # world-level setup
    def add_swarm(self, swarm):
        self.swarms[swarm.torrent_id] = swarm

    def compute_link_capacity(self, agent_a, agent_b):
        """radio capacity as a function of distance"""
        dx = agent_a.position[0] - agent_b.position[0]
        dy = agent_a.position[1] - agent_b.position[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # path-loss attenuation
        capacity = self.c_max / (1.0 + (dist/self.d0)**self.gamma)
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

    # per-channel chunk capacity following Fan et al/
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

        self.recalculate_bandwidth()

    def finish_transfer(self, transfer):
        t_id = transfer.torrent_id
        downloader = transfer.downloader

        self.active_transfers.remove(transfer)
        downloader.active_downloads.remove(transfer)
        transfer.uploader.active_uploads.remove(transfer)

        downloader.downloading_pieces[t_id].remove(transfer.piece_id)
        downloader.completed_pieces[t_id].add(transfer.piece_id)

        # record metrics
        downloader.record_useful_chunks(self,t_id)
        self.recalculate_bandwidth()

        self.schedule(0.0, "PICK_PIECE", downloader, data=t_id)
        # wake up any idle neighbours in this swarm (Fixes cascading pipeline stall)
        for neighbours in downloader.neighbours[t_id]:
            if not any(t.torrent_id == t_id for t in neighbours.active_downloads):
                self.schedule(0.0, "PICK_PIECE", neighbours, data=t_id)

    def append_pieces(self, torrent_id, count, publisher_agent):
        """Append new pieces to the end of the seed (Append-only Mutable Torrent)"""
        swarm = self.swarms[torrent_id]
        new_start = swarm.published_pieces
        swarm.published_pieces += count
        new_pieces = set(range(new_start, swarm.published_pieces))

        publisher_agent.completed_pieces[torrent_id].update(new_pieces)
        print(f"[Time {self.time:6.2f}s] APPEND: Swarm '{torrent_id}' +{count} pieces added. "
              f"New stream horizon: [0 .. {swarm.published_pieces - 1}]")

        # wake up idle leechers
        for agent in swarm.participants:
            if agent != publisher_agent: 
                self.schedule(0.0, "PICK_PIECE", agent, data=torrent_id)

    def finalize_stream(self, torrent_id):
        """Mark the stream as complete"""
        swarm = self.swarms[torrent_id]
        swarm.is_finalized = True
        print(f"[Time {self.time:6.2f}s] FINALIZE: Swarm '{torrent_id}': Closed at {swarm.published_pieces} total pieces.")

        # Trigger idle leechers to evaluate total completion
        for agent in swarm.participants:
            if not any(t.torrent_id == torrent_id for t in agent.active_downloads):
                self.schedule(0.0, "PICK_PIECE", agent, data=torrent_id)
        
    def run(self):
        while self.event_queue:
            self.time, _, event_type, agent, data = heapq.heappop(self.event_queue)

            if event_type == "PICK_PIECE":
                torrent_id = data
                agent.pick_next_piece(self, torrent_id)

            elif event_type == "APPEND_PIECES":
                torrent_id, count = data
                self.append_pieces(torrent_id, count, agent)

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
    def __init__(self, agent_id, position=(0.0, 0.0), download_speed_mbps=10, upload_speed_mbps=10):
        self.agent_id = agent_id
        self.position = position # (x,y)
        self.download_speed = download_speed_mbps
        self.upload_speed = upload_speed_mbps

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

    def join_swarm(self, swarm, strategy="rarest_random", hybrid_s=0.5, segment_k=5, upstream_peer=None, is_seeder=False):
        t_id = swarm.torrent_id
        swarm.participants.add(self)

        self.completed_pieces[t_id] = set(range(swarm.published_pieces)) if is_seeder else set()
        self.downloading_pieces[t_id] = set()
        self.neighbours[t_id] = []
        self.strategies[t_id] = {
            "strategy": strategy,
            "hybrid_s": hybrid_s,
            "segment_k": segment_k
        }
        self.upstream_peers[t_id] = upstream_peer

        self.start_time[t_id] = None
        self.finish_time[t_id] = None
        self.u_x_history[t_id] = []
        self.r_bar_snapshots[t_id] = []

    def connect(self, other_agent, torrent_id):
        if other_agent not in self.neighbours[torrent_id]:
            self.neighbours[torrent_id].append(other_agent)
        if self not in other_agent.neighbours[torrent_id]:
            other_agent.neighbours[torrent_id].append(self)

    def record_useful_chunks(self, sim, torrent_id):
        """U(x) per Eq. 10 in Fan et al."""
        completed = self.completed_pieces[torrent_id]
        x = len(completed)
        contiguous_len = 0
        while contiguous_len in completed:
            contiguous_len += 1

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
                r_i_sum += sum(1 for n in self.neighbours[torrent_id] if p in n.completed_pieces[torrent_id])
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
            if len(self.completed_pieces[torrent_id]) == swarm.published_pieces and swarm.is_finalized and self.finish_time is None:
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
        neighbours = self.neighbours[torrent_id]

        # RAREST-RANDOM
        if strategy == "rarest_random":
            available_pieces = Counter()
            peer_map = {} # piece_id -> list of peers who have it
            for n in neighbours:
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
            for n in neighbours:
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
            # default to first connected neighbour if None
            if upstream is None and len(neighbours) > 0:
                upstream = neighbours[0]
                self.upstream_peers[torrent_id] = upstream

            if upstream is not None:
                available = upstream.completed_pieces[torrent_id] & missing
                if available:
                    chosen_piece = min(available)
                    target_agent = upstream

        # HYBRID (sequential w/ prob s, random w/ prob 1-s)
        elif strategy == 'hybrid':
            available_pieces = Counter()
            peer_map = {}
            for n in neighbours:
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
            for n in neighbours:
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
    sim = TorrentSim(c_max=10.0, d0=10.0, gamma=2.0)

    # Instantiate Swarm
    swarm_vod = Swarm("Movie_Stream", initial_pieces=6, piece_size_mb=1)
    sim.add_swarm(swarm_vod)

    # Create 4 Spatial Robot Agents positioned along the X-axis
    agents = [
        Agent(agent_id=0, position=(0.0, 0.0)),   # Seeder at origin
        Agent(agent_id=1, position=(5.0, 0.0)),   # 5m away
        Agent(agent_id=2, position=(15.0, 0.0)),  # 15m away
        Agent(agent_id=3, position=(30.0, 0.0))   # 30m away
    ]
    sim.agents = agents

    # Join Swarm in Cascading chain: 0 -> 1 -> 2 -> 3
    agents[0].join_swarm(swarm_vod, is_seeder=True)
    agents[1].join_swarm(swarm_vod, strategy="rarest_random", upstream_peer=agents[0])
    agents[2].join_swarm(swarm_vod, strategy="rarest_random", upstream_peer=agents[1])
    agents[3].join_swarm(swarm_vod, strategy="rarest_random", upstream_peer=agents[2])

    agents[0].connect(agents[1], "Movie_Stream")
    agents[1].connect(agents[2], "Movie_Stream")
    agents[2].connect(agents[3], "Movie_Stream")

    # Schedule initial pick events
    for a in [agents[1], agents[2], agents[3]]:
        sim.schedule(0.0, "PICK_PIECE", a, data="Movie_Stream")

    # Dynamic Appends
    sim.schedule(5.0, "APPEND_PIECES", agents[0], data=("Movie_Stream", 4))
    sim.schedule(10.0, "FINALIZE_STREAM", agents[0], data="Movie_Stream")

    print("=" * 80)
    print("SPATIAL ROBOT RADIO SIMULATION TRACE")
    print("=" * 80)

    # Print initial distance and pairwise channel capacities
    for i in range(len(agents) - 1):
        u, v = agents[i], agents[i+1]
        dist, cap = sim.compute_link_capacity(u, v)
        print(f"Link Agent {u.agent_id} <-> Agent {v.agent_id}: Dist = {dist:4.1f}m | Max Link Rate = {cap:5.2f} Mbps")
    print("-" * 80)

    sim.run()

    print("\n" + "=" * 80)
    print(f"{'Agent ID':<10} | {'Position (x,y)':<18} | {'Throughput (T)':<15} | {'Sequentiality (S)':<18} | {'Robustness (R)':<18}")
    print("-" * 80)

    for agent in agents:
        if agent.start_time.get("Movie_Stream") is not None and agent.finish_time.get("Movie_Stream") is not None:
            t, s, r = agent.get_metrics(sim, "Movie_Stream")
            pos_str = f"({agent.position[0]:.1f}, {agent.position[1]:.1f})"
            print(f"Agent {agent.agent_id:<4} | {pos_str:<18} | {t:6.2f} Mbps       | {s:6.4f}     |   {r:6.4f}")