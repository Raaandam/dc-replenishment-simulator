"""FIFO batch inventory with expiry, used by both stores and the DC."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Batch:
    qty: float
    expires_at: float  # simulation hour


@dataclass
class Stock:
    batches: deque = field(default_factory=deque)
    on_order: float = 0.0

    @property
    def on_hand(self) -> float:
        return sum(b.qty for b in self.batches)

    @property
    def position(self) -> float:
        """Inventory position = on hand + on order (what a proper policy reviews)."""
        return self.on_hand + self.on_order

    def receive(self, qty: float, expires_at: float) -> None:
        if qty > 0:
            self.batches.append(Batch(qty, expires_at))

    def take(self, qty: float) -> float:
        """Issue oldest stock first; return the quantity actually issued."""
        issued = 0.0
        while qty > 0 and self.batches:
            b = self.batches[0]
            used = min(b.qty, qty)
            b.qty -= used
            qty -= used
            issued += used
            if b.qty <= 0:
                self.batches.popleft()
        return issued

    def take_batches(self, qty: float) -> list[Batch]:
        """Issue oldest stock first, keeping expiry dates (for DC -> store transfers)."""
        out = []
        while qty > 0 and self.batches:
            b = self.batches[0]
            used = min(b.qty, qty)
            out.append(Batch(used, b.expires_at))
            b.qty -= used
            qty -= used
            if b.qty <= 0:
                self.batches.popleft()
        return out

    def expire(self, now: float) -> float:
        """Write off expired batches; return the wasted quantity."""
        wasted = 0.0
        while self.batches and self.batches[0].expires_at <= now:
            wasted += self.batches.popleft().qty
        # Batches arrive in expiry order per SKU, but check the rest too
        keep = deque()
        for b in self.batches:
            if b.expires_at <= now:
                wasted += b.qty
            else:
                keep.append(b)
        self.batches = keep
        return wasted
