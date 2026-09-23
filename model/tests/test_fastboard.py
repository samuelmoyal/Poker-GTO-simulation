import os
import random
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import cards, equity, fastboard, poker  # noqa: E402


@unittest.skipUnless(fastboard.available(), "boardlib not built (cd tools/boardlib && cargo build --release)")
class FastBoardTest(unittest.TestCase):
    def test_strength_is_identical_to_numpy(self):
        rng = random.Random(0)
        for k in (5, 6, 7):
            hands = [rng.sample(range(52), k) for _ in range(30000)]
            np.testing.assert_array_equal(fastboard.strength(np.array(hands)), poker.strength(np.array(hands)))
        for ranks in ([0, 1, 2, 3, 12], [7, 8, 9, 10, 11, 12], [4, 5, 6], [0, 12, 11, 1]):   # many pairs / straights / flushes
            deck = [4 * r + s for r in ranks for s in range(4)]
            for k in (5, 6, 7):
                if len(deck) >= k:
                    hands = np.array([rng.sample(deck, k) for _ in range(4000)])
                    np.testing.assert_array_equal(fastboard.strength(hands), poker.strength(hands))
        deck = [4 * r + s for r in range(13) for s in (0, 1)]                                   # two suits: flushes, straight flushes
        hands = np.array([rng.sample(deck, 7) for _ in range(8000)])
        np.testing.assert_array_equal(fastboard.strength(hands), poker.strength(hands))

    def test_equity_and_features_match_the_win_table(self):
        boards = [["Td", "9d", "6h", "Qc"], ["As", "Ks", "Qs", "2d"], ["5h", "5d", "5c", "9s"]]
        fb = fastboard.FastBoards(boards)
        rng = np.random.default_rng(0)
        cpu = torch.device("cpu")
        for j, b in enumerate(boards):
            tb = equity.BoardTables(b, cpu)
            np.testing.assert_array_equal(fb.cat6[j], tb.cat6)
            np.testing.assert_array_equal(fb.outs[j], tb.outs)
            np.testing.assert_array_equal(fb.hand_ok[j], tb.hand_ok.numpy())
            ranges = rng.random((4, cards.NUM_COMBOS)).astype(np.float32) * (rng.random((4, cards.NUM_COMBOS)) < 0.5)
            ranges[3] = 0.0
            ranges[3, :5] = 1.0                        # a tiny range: many hands have Z = 0 -> 0.5
            ranges = ranges * tb.hand_ok.numpy()       # as in real use: no range mass on hands the board blocks
            want = np.stack([tb.equity(torch.tensor(r)).numpy() for r in ranges])
            got = fb.equity(np.tile(ranges[None], (len(boards), 1, 1)))[j]
            self.assertLess(float(np.abs(got - want).max()), 2e-5, f"board {b}")
            np.testing.assert_allclose(fb.uniform_equity()[j], tb.uniform_equity().numpy(), atol=2e-5)
        wd = fb.win_tables()                           # the Rust table equals the torch one entry for entry
        for j, b in enumerate(boards):
            np.testing.assert_allclose(wd[j], equity.BoardTables(b, cpu).WD.numpy(), atol=1e-6, err_msg=f"WD {b}")


if __name__ == "__main__":
    unittest.main()
