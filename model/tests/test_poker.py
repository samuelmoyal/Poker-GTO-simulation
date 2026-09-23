import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import cards, poker  # noqa: E402
from gto.cards import evaluate  # noqa: E402


def dense(values):
    order = {v: i for i, v in enumerate(sorted(set(values)))}
    return [order[v] for v in values]


class PokerTest(unittest.TestCase):
    def check(self, hands):
        got = poker.strength(np.array(hands))
        ref = [evaluate([cards.DECK[c] for c in h]) for h in hands]
        self.assertEqual(dense(list(got)), dense(ref))  # same order AND same ties
        self.assertEqual(list(poker.category(got)), [t[0] for t in ref])

    def test_random_hands(self):
        rng = random.Random(0)
        for k in (5, 6, 7):
            self.check([rng.sample(range(52), k) for _ in range(6000)])

    def test_low_diversity_hands(self):
        """few ranks / few suits: many pairs, trips, quads, full houses, flushes, straight flushes"""
        rng = random.Random(1)
        pool_r = [[0, 1, 2, 3, 12], [7, 8, 9, 10, 11, 12], [4, 5, 6], [0, 12, 11, 1]]
        for ranks in pool_r:
            deck = [4 * r + s for r in ranks for s in range(4)]
            for k in (5, 6, 7):
                self.check([rng.sample(deck, k) for _ in range(1500) if len(deck) >= k])
        for suits in ([0, 1], [3], [0, 1, 2]):
            deck = [4 * r + s for r in range(13) for s in suits]
            self.check([rng.sample(deck, 7) for _ in range(3000)])

    def test_known_hands(self):
        c = cards.CARD_ID
        h = lambda *x: [c[i] for i in x]  # noqa: E731
        wheel = poker.strength(np.array([h("Ah", "2d", "3c", "4s", "5h", "Kd", "Kc")]))[0]
        six = poker.strength(np.array([h("2h", "3d", "4c", "5s", "6h", "Kd", "Kc")]))[0]
        self.assertLess(wheel, six)  # the wheel loses to a six-high straight
        two_trips = poker.strength(np.array([h("Ah", "Ad", "Ac", "Ks", "Kh", "Kd", "2c")]))[0]
        self.assertEqual(poker.category(two_trips), 6)  # AAA KK (second trips plays as the pair)
        steel = poker.strength(np.array([h("As", "2s", "3s", "4s", "5s", "Kd", "Kc")]))[0]
        self.assertEqual(poker.category(steel), 8)


if __name__ == "__main__":
    unittest.main()
