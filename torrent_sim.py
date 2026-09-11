import heapq
import random
from collections import Counter

import pdb

class Transfer:
    def __init__(self, transfer_id, downloader, uploader, piece_id, total_size_mb):
        self.transfer_id = transfer_id
        self.downloader = downloader
        self.uploader = uploader
        self.piece_id = piece_id
        self.remaining_mb = total_size_mb
        self.current_rate_mbps = 0.0
        self.last_update_time = 0.0
        self.scheduled_finish_time = float('inf')
        
class TorrentSim:
    def __init__(self, num_pieces=20, piece_size_mb=1):
        self.num_pieces = num_pieces
        self.piece_size_mb = piece_size_mb
        self.time = 0.0
        self.counter = 0  # Unique tie-breaker sequence number
        self.transfer_counter = 0
        self.event_queue = []  # (time, event_type, peer, data)
        self.active_transfers = set()

    def schedule(self, delay, event_type, peer, data=None):
        self.counter += 1
        heapq.heappush(
            self.event_queue, 
            (self.time + delay, self.counter, event_type, peer, data)
        )

    def _update_transfer_progress(self):
        """Advance bytes transferred for active streams up to this sim time"""
        for t in self.active_transfers:
            elapsed = self.time - t.last_update_time
            if elapsed > 0 and t.current_rate_mbps > 0:
                mb_transferred = (t.current_rate_mbps * elapsed) / 8.0
                t.remaining_mb = max(0.0, t.remaining_mb - mb_transferred)
            t.last_update_time = self.time

    def recalculate_bandwidth(self):
        """
        Dynamically divide uploader/downloader capcaity among active streams per Fan et al. per-chunk capacity formulation
        """
        self._update_transfer_progress()

        for t in self.active_transfers:
            # uploader capacity split uniformly across uploads
            uploader_share = t.uploader.upload_speed / len(t.uploader.active_uploads)
            downloader_share = t.downloader.download_speed / len(t.downloader.active_downloads)

            # bottleneck rate for this transfer
            t.current_rate_mbps = min(uploader_share, downloader_share)

            if t.current_rate_mbps > 0:
                remaining_bits = t.remaining_mb * 8.0
                time_to_complete = remaining_bits / t.current_rate_mbps
                t.scheduled_finish_time = self.time + time_to_complete

                # push updated finish time to event queue
                self.schedule(time_to_complete, "PIECE_COMPLETE", t.downloader, data=t)

    def start_transfer(self, downloader, uploader, piece_id):
        self.transfer_counter += 1
        transfer = Transfer(self.transfer_counter, downloader, uploader, piece_id, self.piece_size_mb)

        downloader.downloading_pieces.add(piece_id)
        downloader.active_downloads.add(transfer)
        uploader.active_uploads.add(transfer)
        self.active_transfers.add(transfer)

        self.recalculate_bandwidth()

    def finish_transfer(self, transfer):
        self.active_transfers.remove(transfer)
        transfer.downloader.active_downloads.remove(transfer)
        transfer.uploader.active_uploads.remove(transfer)
        transfer.downloader.downloading_pieces.remove(transfer.piece_id)
        transfer.downloader.completed_pieces.add(transfer.piece_id)

        # record metrics
        transfer.downloader.record_useful_chunks(self)

        # wake up idle neighbours now that a new piece is available
        for neighbour in transfer.downloader.neighbours:
            if not neighbour.active_downloads:
                self.schedule(0.0, "PICK_PIECE", neighbour)

        self.recalculate_bandwidth()

    def run(self):
        while self.event_queue:
            self.time, _, event_type, peer, data = heapq.heappop(self.event_queue)
            if event_type == "PICK_PIECE":
                peer.pick_next_piece(self)
            elif event_type == "PIECE_COMPLETE":
                transfer = data

                # filter stale events
                if abs(self.time - transfer.scheduled_finish_time) > 1e-7:
                    continue

                if transfer in self.active_transfers:
                    self.finish_transfer(transfer)
                    print(f"[Time {self.time:6.2f}s] Peer {peer.peer_id:2d} got piece {transfer.piece_id:2d} "
                          f"from Peer {transfer.uploader.peer_id:2d} "
                          f"({len(peer.completed_pieces)}/{self.num_pieces} complete)")

                    if len(peer.completed_pieces) == self.num_pieces:
                        peer.finish_time = self.time
                    else:
                        # Immediately attempt to request next piece
                        self.schedule(0.0, "PICK_PIECE", peer)

class Peer:
    def __init__(self, peer_id, download_speed_mbps=10, upload_speed_mbps=10, strategy="rarest_random", upstream_peer=None):
        self.peer_id = peer_id
        self.download_speed = download_speed_mbps
        self.upload_speed = upload_speed_mbps
        self.strategy = strategy # "rarest_random", "sequential", or "cascading"
        self.completed_pieces = set()
        self.downloading_pieces = set()
        self.neighbours = []

        # continuous time tracking sets
        self.active_uploads = set()
        self.active_downloads = set()
        self.upstream_peer = None

        # metrics
        self.start_time = None
        self.finish_time = None
        self.u_x_history = [] # history of useful chunks
        self.r_bar_snapshots = [] # history of robustness

    def connect(self, other_peer):
        if other_peer not in self.neighbours:
            self.neighbours.append(other_peer)
        if self not in other_peer.neighbours:
            other_peer.neighbours.append(self)

    def record_useful_chunks(self, sim):
        """U(x) per Eq. 10 in Fan et al."""
        x = len(self.completed_pieces)
        contiguous_len = 0
        while contiguous_len in self.completed_pieces:
            contiguous_len += 1

        u_x = contiguous_len / x if x > 0 else 0.0
        self.u_x_history.append(u_x)

        r_bar = self.compute_current_r_bar(sim)
        self.r_bar_snapshots.append(r_bar)

    def compute_current_r_bar(self, sim):
        """r_bar per Eq. 8 in Fan et al."""
        r_i_sum = 0
        for p in range(sim.num_pieces):
            if self.strategy == "cascading":
                if self.upstream_peer and p in self.upstream_peer.completed_pieces:
                    r_i_sum += 1
            else:
                r_i_sum += sum(1 for n in self.neighbours if p in n.completed_pieces)
        return r_i_sum / sim.num_pieces

    def get_metrics(self, sim, p_error=0.5):
        """Return Throughput [Mbps], Sequentiality [0..1], Robustness [0..1]"""
        if self.start_time is None or self.finish_time is None:
            return 0.0,0.0,0.0

        # throughput
        duration = self.finish_time - self.start_time
        file_size_bits = sim.num_pieces * sim.piece_size_mb * 8.0
        throughput = file_size_bits / duration if duration > 0 else 0.0

        # sequentiality (eq. 10)
        sequentiality = sum(self.u_x_history) / sim.num_pieces if self.u_x_history else 0.0

        # robustness (eq. 8)
        avg_r_bar = sum(self.r_bar_snapshots) / len(self.r_bar_snapshots) if self.r_bar_snapshots else 0.0
        robustness = 1.0 - (p_error ** avg_r_bar)

        return throughput, sequentiality, robustness

    def pick_next_piece(self, sim):
        if self.start_time is None:
            self.start_time = sim.time

        # prevent overlapping transfers
        if self.active_downloads:
            return

        missing = set(range(sim.num_pieces)) - self.completed_pieces - self.downloading_pieces
        if not missing:
            return

        chosen_piece = None
        target_peer = None

        # RAREST-RANDOM
        if self.strategy == "rarest_random":
            available_pieces = Counter()
            peer_map = {} # piece_id -> list of peers who have it
            for n in self.neighbours:
                for p in n.completed_pieces:
                    if p in missing:
                        available_pieces[p] += 1
                        peer_map.setdefault(p, []).append(n)

            if available_pieces:
                min_freq = min(available_pieces.values())
                rarest_candidates = [p for p, count in available_pieces.items() if count == min_freq]
                chosen_piece = random.choice(rarest_candidates)
                target_peer = random.choice(peer_map[chosen_piece])

        # SEQUENTIAL
        elif self.strategy == "sequential":
            available_pieces = set()
            peer_map = {}
            for n in self.neighbours:
                for p in n.completed_pieces:
                    if p in missing:
                        available_pieces.add(p)
                        peer_map.setdefault(p, []).append(n)

            if available_pieces:
                chosen_piece = min(available_pieces)
                target_peer = random.choice(peer_map[chosen_piece])

        # CASCADING
        elif self.strategy == "cascading":
            # check if current bound target peer still has missing pieces to offer
            if self.upstream_peer is not None:
                available = self.upstream_peer.completed_pieces & missing
                if available:
                    chosen_piece = min(available)
                    target_peer = self.upstream_peer

        # dispatch task if a valid piece and peer target were selected
        if chosen_piece is not None and target_peer is not None:
            sim.start_transfer(
                downloader=self,
                uploader=target_peer,
                piece_id=chosen_piece
            )

def run_benchmark(strategy_name):
    sim = TorrentSim(num_pieces=20, piece_size_mb=1)  # 20 MB total file

    # Seed
    seeder = Peer(peer_id=0, upload_speed_mbps=10)
    seeder.completed_pieces = set(range(20))

    # 4 Leechers
    leechers = []
    prev_peer = seeder
    for i in range(1, 5):
        p = Peer(peer_id=i, strategy=strategy_name, download_speed_mbps=10, upload_speed_mbps=10)
        
        if strategy_name == "cascading":
            p.upstream_peer = prev_peer
            p.connect(prev_peer)
            prev_peer = p
        else:
            seeder.connect(p)
            for existing in leechers:
                existing.connect(p)
                
        leechers.append(p)

    sim.peers = leechers

    for p in leechers:
        sim.schedule(0.0, "PICK_PIECE", p)

    sim.run()

    # Collect average metrics across leechers
    t_avg = sum(p.get_metrics(sim)[0] for p in leechers) / len(leechers)
    s_avg = sum(p.get_metrics(sim)[1] for p in leechers) / len(leechers)
    r_avg = sum(p.get_metrics(sim)[2] for p in leechers) / len(leechers)

    return t_avg, s_avg, r_avg

if __name__=="__main__":
    # Execute comparisons
    results = {}
    strats = ["rarest_random", "sequential", "cascading"]
    for strat in strats:
        print(f"STRAT: {strat}")
        results[strat] = run_benchmark(strat)

    print(f"{'Strategy':<15} | {'Throughput (T)':<15} | {'Sequentiality (S)':<18} | {'Robustness (R)':<15}")
    print("-" * 72)
    for strat, (t, s, r) in results.items():
        print(f"{strat:<15} | {t:6.2f} Mbps       | {s:6.4f}             | {r:6.4f}")