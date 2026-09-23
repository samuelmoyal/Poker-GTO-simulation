import glob
import os
import sys
import unittest

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import cards, equity, paths  # noqa: E402
from gto.cards import evaluate  # noqa: E402

BOARD = ["Td", "9d", "6h", "Qc"]
SHARDS = sorted(glob.glob(os.path.join(paths.TURN_LABELS_DIR, "part-*.parquet")))


class EquityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = equity.BoardTables(BOARD)

    def brute(self, a, b):
        wins = n = 0.0
        for c in cards.DECK:
            if c in BOARD or c in (a[:2], a[2:], b[:2], b[2:]):
                continue
            ea, eb = evaluate([a[:2], a[2:], *BOARD, c]), evaluate([b[:2], b[2:], *BOARD, c])
            wins += 1.0 if ea > eb else 0.5 if ea == eb else 0.0
            n += 1
        return wins / n

    def test_pairwise_equity_matches_brute_force(self):
        for a, b in [("AsAh", "KsKh"), ("JhTh", "AcKc"), ("8h7h", "QdQs"), ("5s4s", "AhAd"), ("KdJd", "9c9s")]:
            self.assertAlmostEqual(float(self.t.W[cards.combo_index(a), cards.combo_index(b)]), self.brute(a, b), places=5)

    def test_symmetry_and_zero_sum(self):
        W, D = self.t.W, equity._const(self.t.device)["disjoint"]
        ok = (D > 0) & self.t.hand_ok[:, None] & self.t.hand_ok[None, :]
        self.assertLess(float(((W + W.T - 1).abs() * ok).max()), 1e-5)  # equity(h,h') + equity(h',h) = 1

    def test_uniform_equity_range(self):
        eq = self.t.uniform_equity().cpu().numpy()[self.t.hand_ok.cpu().numpy()]
        self.assertTrue(0.0 <= eq.min() and eq.max() <= 1.0)
        self.assertGreater(float(self.t.uniform_equity()[cards.combo_index("QdQs")]), 0.8)  # top set

    def test_corrected_marginals_sum(self):
        rng = np.random.default_rng(0)
        r1 = torch.tensor(rng.random(1326) * ~np.isin(cards.COMBO_CARDS, [cards.CARD_ID[c] for c in BOARD]).any(1), dtype=torch.float32)
        r2 = torch.tensor(rng.random(1326) * ~np.isin(cards.COMBO_CARDS, [cards.CARD_ID[c] for c in BOARD]).any(1), dtype=torch.float32)
        c1, c2 = equity.corrected_marginals(r1, r2), equity.corrected_marginals(r2, r1)
        self.assertAlmostEqual(float(c1.sum()), 1.0, places=5)
        # with the corrected marginals, equity is exactly zero-sum: sum c1*eq1 + sum c2*eq2 = 1
        eq1, eq2 = self.t.equity(r2).cpu(), self.t.equity(r1).cpu()
        self.assertAlmostEqual(float((c1 * eq1).sum() + (c2 * eq2).sum()), 1.0, places=4)


@unittest.skipUnless(SHARDS and "in_oop" in pq.read_schema(SHARDS[0]).names, "needs a schema-v2 label shard")
class SolverAgreementTest(unittest.TestCase):
    def test_marginals_match_solver_and_labels_are_zero_sum(self):
        t = pq.read_table(SHARDS[0]).slice(0, 25).to_pydict()
        for i in range(25):
            i_oop, i_ip = (torch.tensor(t[k][i], dtype=torch.float32) for k in ("in_oop", "in_ip"))
            c_oop, c_ip = equity.corrected_marginals(i_oop, i_ip), equity.corrected_marginals(i_ip, i_oop)
            np.testing.assert_allclose(c_oop.numpy(), t["r_oop"][i], atol=2e-5)  # my formula == the solver's weights
            np.testing.assert_allclose(c_ip.numpy(), t["r_ip"][i], atol=2e-5)


if __name__ == "__main__":
    unittest.main()
