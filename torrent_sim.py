import heapq
import random
from collections import Counter

class TorrentSim:
    def __init__(self, num_pieces=100, piece_size_mb=2):
        self.num_pieces = num_pieces
        self.piece_size_mb = piece_size_mb
        self.time = 0.0
        self.counter = 0  # Unique tie-breaker sequence number
        self.event_queue = []  # (time, event_type, peer, data)
        self.peers = []

    def schedule(self, delay, event_type, peer, data=None):
        self.counter += 1
        heapq.heappush(
            self.event_queue, 
            (self.time + delay, self.counter, event_type, peer, data)
        )

    def run(self):
        while self.event_queue:
            self.time, _, event_type, peer, data = heapq.heappop(self.event_queue)
            if event_type == "PICK_PIECE":
                peer.pick_next_piece(self)
            elif event_type == "PIECE_COMPLETE":
                peer.on_piece_received(self, piece_id=data)

class Peer:
    def __init__(self, peer_id, download_speed_mbps=10, upload_speed_mbps=10, strategy="rarest_random", upstream_peer=None):
        self.peer_id = peer_id
        self.download_speed = download_speed_mbps
        self.upload_speed = upload_speed_mbps
        self.strategy = strategy # "rarest_random", "sequential", or "cascading"
        self.completed_pieces = set()
        self.downloading_pieces = set()
        self.neighbours = []

        self.upstream_peer = None

    def connect(self, other_peer):
        if other_peer not in self.neighbours:
            self.neighbours.append(other_peer)
        if self not in other_peer.neighbours:
            other_peer.neighbours.append(self)

    def pick_next_piece(self, sim):
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
                available_from_upstream = self.upstream_peer.completed_pieces & missing
                if not available_from_upstream:
                    chosen_piece = min(available_from_upstream)
                    target_peer = self.upstream_peer

        # dispatch tash if a valid piece and peer target were selected
        if chosen_piece is not None and target_peer is not None:
            self.downloading_pieces.add(chosen_piece)
        
            # Calculate simulated latency/transfer time without actual file payload
            effective_speed = min(self.download_speed, 10.0)  # assumes neighbor cap
            transfer_time = (sim.piece_size_mb * 8) / effective_speed
            sim.schedule(transfer_time, "PIECE_COMPLETE", self, data=chosen_piece)

    def on_piece_received(self, sim, piece_id):
        self.downloading_pieces.remove(piece_id)
        self.completed_pieces.add(piece_id)
        print(f"[Time {sim.time:.2f}s] Peer {self.peer_id} finished piece {piece_id} ({len(self.completed_pieces)}/{sim.num_pieces})")
        
        # Immediately pick the next piece
        sim.schedule(0.0, "PICK_PIECE", self)


# Simulation Setup
sim = TorrentSim(num_pieces=20, piece_size_mb=1)

# Seeders with complete files
seeder_1 = Peer(peer_id=0, upload_speed_mbps=10)
seeder_1.completed_pieces = set(range(10))

seeder_2 = Peer(peer_id=1, upload_speed_mbps=10)
seeder_2.completed_pieces = {0, 1, 2, 3, 4}  # Partial seed

# Leechers testing different policies
leecher_rr = Peer(peer_id=2, strategy="rarest_random", download_speed_mbps=10)
leecher_seq = Peer(peer_id=3, strategy="sequential", download_speed_mbps=10)
leecher_cas = Peer(peer_id=4, strategy="cascading", download_speed_mbps=10)

# Topology configuration
for leecher in [leecher_rr, leecher_seq, leecher_cas]:
    seeder_1.connect(leecher)
    seeder_2.connect(leecher)

# Start event loops
for leecher in [leecher_rr, leecher_seq, leecher_cas]:
    sim.schedule(0.0, "PICK_PIECE", leecher)

sim.run()