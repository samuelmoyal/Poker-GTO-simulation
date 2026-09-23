import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import cards, flopdump, labels, nativeflops, paths, states  # noqa: E402

FILES = states.flop_files()


@unittest.skipUnless(FILES, "no flop solves cached")
class FlopReplayTest(unittest.TestCase):
    def test_replay_invariants(self):
        path = FILES[0]
        matchup, flop, _ = flopdump.parse_name(path)
        lines = flopdump.replay(flopdump.load(path), matchup, flop)
        root = flopdump.root_reach(matchup, flop)
        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("x-x", [l.line for l in lines])
        pot0 = next(l.pot for l in lines if l.line == "x-x")
        for l in lines:
            self.assertGreaterEqual(l.pot, pot0)
            self.assertGreaterEqual(l.stack, 0)
            self.assertTrue(0 < l.mass <= 1.0 + 1e-9)
            for side in (0, 1):
                for h, r in l.reach[side].items():
                    self.assertLessEqual(r, root[side][h] + 1e-9)
        # lines are mutually exclusive and the players' hands independent, so line probabilities add up to <= 1
        # (the rest is folds)
        self.assertLessEqual(sum(l.mass for l in lines), 1.0 + 1e-6)
        # pot accounting: after the last bet/raise-to N is called, each player has put N in
        for l in lines:
            amounts = [int(a[1:]) for a in l.line.split("-") if a[0] in "br"]
            self.assertEqual(l.pot, pot0 + 2 * (amounts[-1] if amounts else 0), l.line)

    def test_states(self):
        sts = states.flop_states(FILES[0], 12, seed=1)
        self.assertGreater(len(sts), 0)
        again = states.flop_states(FILES[0], 12, seed=1)
        self.assertEqual([s.id for s in sts], [s.id for s in again])  # deterministic
        for s in sts:
            self.assertEqual(len(s.board), 4)
            self.assertEqual(s.split, cards.split_of(s.board))
            for r in (s.oop, s.ip):
                self.assertAlmostEqual(max(r.values()), 1.0)
                self.assertTrue(all(not ({cards.CARD_ID[c] for c in s.board} & set(cards.parse_hand(h))) for h in r))

    def test_no_split_leakage(self):
        seen = {}
        for path in FILES:
            for s in states.flop_states(path, 4, seed=0):
                key = cards.canonical_flop_key(s.board)
                self.assertEqual(seen.setdefault(key, s.split), s.split)


@unittest.skipUnless(FILES and os.path.exists(paths.LABEL_BIN), "needs flops and the built turn-labels binary")
class LabelPipelineTest(unittest.TestCase):
    def test_solve_and_store(self):
        sts = states.flop_states(FILES[0], 3, seed=2)
        rows = []
        with labels.LabelSolver() as solver:
            for s in sts:
                resp = solver.solve(s.request(1.0))
                self.assertTrue(resp["ok"], resp)
                rows.append(labels.to_row(s, resp))
        for r in rows:
            # constant-sum game: range-weighted EVs (fraction of pot) add up to 1
            self.assertAlmostEqual(float(r["r_oop"] @ r["v_oop"] + r["r_ip"] @ r["v_ip"]), 1.0, places=4)
            self.assertAlmostEqual(float(r["r_oop"].sum()), 1.0, places=4)
            root = r["root_strategy"].reshape(labels.MAX_ROOT_ACTIONS, -1)
            present = r["r_oop"] > 0
            np.testing.assert_allclose(root[:, present].sum(0), 1.0, atol=1e-3)  # strategies are distributions
        with tempfile.TemporaryDirectory() as tmp:
            labels.write_shard(rows, tmp)
            self.assertEqual(labels.existing_ids(tmp), {r["id"] for r in rows})


NATIVE = [p for p in FILES if p.endswith(f"__{nativeflops.PID}.json.gz")]


@unittest.skipUnless(NATIVE, "no native flop solves yet")
class NativeFlopTest(unittest.TestCase):
    def test_lines(self):
        path = NATIVE[0]
        matchup, flop, _ = flopdump.parse_name(path)
        lines = nativeflops.load_lines(path)
        root = flopdump.root_reach(matchup, flop)
        pot0 = next(l.pot for l in lines if l.line == "x-x")
        self.assertLessEqual(sum(l.mass for l in lines), 1.0 + 1e-6)
        for l in lines:
            amounts = [int(a[1:]) for a in l.line.split("-") if a[0] in "bra"]
            self.assertEqual(l.pot, pot0 + 2 * (amounts[-1] if amounts else 0), l.line)
            for side in (0, 1):
                for h, r in l.reach[side].items():
                    self.assertLessEqual(r, root[side][h] + 1e-6)


if __name__ == "__main__":
    unittest.main()
