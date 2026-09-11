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
    def __init__(self, peer_id, sim, download_speed_mbps=10, upload_speed_mbps=10, strategy="rarest_first"):
        self.peer_id = peer_id
        self.download_speed = download_speed_mbps
        self.upload_speed = upload_speed_mbps
        self.strategy = strategy
        self.completed_pieces = set()
        self.downloading_pieces = set()
        self.neighbors = []

    def connect(self, other_peer):
        self.neighbors.append(other_peer)
        other_peer.neighbors.append(self)

    def pick_next_piece(self, sim):
        missing = set(range(sim.num_pieces)) - self.completed_pieces - self.downloading_pieces
        if not missing:
            return

        # Gather piece availability among unchoked neighbors
        available_pieces = Counter()
        for n in self.neighbors:
            for p in n.completed_pieces:
                if p in missing:
                    available_pieces[p] += 1

        if not available_pieces:
            return

        # Piece selection strategy
        if self.strategy == "rarest_first" and len(self.completed_pieces) > 2:
            min_count = min(available_pieces.values())
            candidates = [p for p, count in available_pieces.items() if count == min_count]
            chosen_piece = random.choice(candidates)
        else:
            # Random First strategy (used at download start to quickly get a piece to seed)
            chosen_piece = random.choice(list(available_pieces.keys()))

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

# Seeder (has all pieces)
seeder = Peer(peer_id=0, sim=sim, upload_speed_mbps=20)
seeder.completed_pieces = set(range(20))

# Leechers
leecher_a = Peer(peer_id=1, sim=sim, strategy="rarest_random")
leecher_b = Peer(peer_id=2, sim=sim, strategy="rarest_random")

seeder.connect(leecher_a)
seeder.connect(leecher_b)
leecher_a.connect(leecher_b)

# Kick off piece picking loops
sim.schedule(0.0, "PICK_PIECE", leecher_a)
sim.schedule(0.0, "PICK_PIECE", leecher_b)

sim.run()