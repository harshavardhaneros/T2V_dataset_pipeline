"""BK-tree for perceptual hash near-duplicate detection."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def phash_to_int(phash: str) -> int:
    return int(phash, 16) if phash else 0


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class BKTree:
    def __init__(self, threshold: int = 8):
        self.threshold = threshold
        self.root: Optional[Tuple[int, str]] = None
        self.children: Dict[Tuple[int, str], Dict[int, Tuple[int, str]]] = {}

    def add(self, value: int, clip_id: str) -> None:
        node = (value, clip_id)
        if self.root is None:
            self.root = node
            self.children[node] = {}
            return
        current = self.root
        while True:
            dist = hamming_distance(value, current[0])
            bucket = self.children.setdefault(current, {})
            if dist == 0:
                return
            if dist in bucket:
                current = bucket[dist]
            else:
                bucket[dist] = node
                self.children[node] = {}
                return

    def find_match(self, value: int) -> Optional[str]:
        if self.root is None:
            return None
        to_visit = [self.root]
        while to_visit:
            node = to_visit.pop()
            dist = hamming_distance(value, node[0])
            if dist <= self.threshold:
                return node[1]
            child_map = self.children.get(node, {})
            for d in range(max(0, dist - self.threshold), dist + self.threshold + 1):
                if d in child_map:
                    to_visit.append(child_map[d])
        return None
