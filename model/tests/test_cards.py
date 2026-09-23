import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import cards  # noqa: E402


class CardsTest(unittest.TestCase):
    def test_combo_indexing(self):
        self.assertEqual(len(cards.COMBOS), 1326)
        for k in (0, 1, 500, 1325):
            self.assertEqual(cards.combo_index(cards.combo_str(k)), k)
        self.assertEqual(cards.combo_index("AhKs"), cards.combo_index("KsAh"))

    def test_blocked_mask(self):
        self.assertEqual(int(cards.blocked_mask(["As"]).sum()), 51)
        self.assertEqual(int(cards.blocked_mask(["As", "Kd"]).sum()), 51 + 51 - 1)

    def test_suit_permutations(self):
        self.assertEqual(len({tuple(p) for p in cards.SUIT_PERMS}), 24)
        rng = np.random.default_rng(0)
        vec = rng.random(1326)
        for perm in cards.SUIT_PERMS:
            cperm = cards.combo_perm(perm)
            self.assertEqual(sorted(cperm), list(range(1326)))  # bijection
            out = cards.permute_combos(vec, cperm)
            self.assertAlmostEqual(out.sum(), vec.sum())
            # a concrete hand: relabel its suits by hand and look it up
            i, j = cards.COMBOS[700]
            a, b = cards.DECK[i], cards.DECK[j]
            na, nb = a[0] + "cdhs"[perm["cdhs".index(a[1])]], b[0] + "cdhs"[perm["cdhs".index(b[1])]]
            self.assertEqual(cperm[700], cards.combo_index(na + nb))
            self.assertEqual(out[cperm[700]], vec[700])
            inv = np.argsort(cperm)
            np.testing.assert_array_equal(cards.permute_combos(out, inv), vec)  # inverse relabelling

    def test_split_is_isomorphism_invariant(self):
        self.assertEqual(cards.split_of(["As", "Ks", "Qh", "2c"]), cards.split_of(["Ad", "Kd", "Qc", "9h"]))
        splits = {cards.split_of([r + "s", "7h", "2d"]) for r in "AKQJT98"}
        self.assertTrue(splits <= {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
