import glob
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import cards, cfr, labels, paths, states  # noqa: E402


class TreeTest(unittest.TestCase):
    def test_lines_and_raise_to(self):
        root = cfr.build_round_tree(pot=55, stack=975, bet_fracs=(0.5,), raise_fracs=(1.0,), max_raises=1)
        ends = sorted(n.line for n in cfr.walk(root) if n.kind == "end")
        self.assertEqual(ends, ["b0.5-c", "b0.5-r1-c", "x-b0.5-c", "x-b0.5-r1-c", "x-x"])
        self.assertEqual(len(ends), 5)  # x-x, x-b-c, b-c, x-b-r-c, b-r-c: the 5 lines of an SRP flop in our solves
        raise_node = next(n for n in cfr.walk(root) if n.kind == "action" and n.line == "b0.5-")
        r = raise_node.children[raise_node.actions.index("r1")]
        self.assertAlmostEqual(r.contrib[1], 27.5 + (55 + 55), places=6)  # raise-to = bet + pot after the call (solver: 139)

    def test_all_in_actions_are_dropped(self):
        root = cfr.build_round_tree(pot=100, stack=120, bet_fracs=(0.5, 1.5), raise_fracs=(1.0,), max_raises=1)
        self.assertEqual(next(cfr.walk(root)).actions, ["x", "b0.5"])  # the 1.5-pot bet would put the stack in


class ClairvoyantGameTest(unittest.TestCase):
    """OOP holds the nuts (share p) or air; IP holds a bluff catcher.  Known equilibrium (pot P, bet B):
    OOP always bets the nuts and bluffs air with  B p / ((1-p)(P+B)),  IP calls with  P / (P+B)."""

    def test_converges_to_the_analytic_equilibrium(self):
        board = ["Ks", "7d", "2c", "9h", "3s"]
        nuts, air, catcher = cards.combo_index("KhKd"), cards.combo_index("6c5c"), cards.combo_index("AhAd")
        p, P, B = 0.3, 100.0, 100.0
        r0, r1 = np.zeros(cfr.H), np.zeros(cfr.H)
        r0[nuts], r0[air], r1[catcher] = p, 1 - p, 1.0
        root = cfr.build_round_tree(pot=P, stack=1000, bet_fracs=(B / P,), raise_fracs=(), max_raises=0, ip_bets=False)
        game = cfr.Cfr(root, r0, r1, P, cfr.Showdown(board, P))
        for _ in range(3000):
            game.iterate()
        s_root = game.avg_sigma(root)                   # OOP: [check, bet]
        bet_nuts, bet_air = s_root[1, nuts], s_root[1, air]
        facing = root.children[1]                       # after a bet: IP [fold, call]
        call = game.avg_sigma(facing)[1, catcher]
        self.assertGreater(bet_nuts, 0.97)
        self.assertAlmostEqual(bet_air, B * p / ((1 - p) * (P + B)), delta=0.04)   # 0.214
        self.assertAlmostEqual(call, P / (P + B), delta=0.04)                      # 0.5
        self.assertLess(game.exploitability() / P, 0.005)                          # < 0.5% of the pot


FLOPS = states.flop_files()


@unittest.skipUnless(FLOPS and os.path.exists(paths.LABEL_BIN), "needs flop solves and the turn-labels binary")
class RiverVsSolverTest(unittest.TestCase):
    def test_river_values_match_postflop_solver(self):
        st = next(s for s in states.flop_states(FLOPS[0], 12, seed=5) if not s.perturbed)
        river = next(c for c in cards.DECK if c not in st.board)
        board5, P, stack = st.board + [river], 100.0, 400.0
        with labels.LabelSolver() as solver:
            resp = solver.solve({"id": "t", "flop": "".join(st.board[:3]), "turn": st.board[3], "river": river, "pot": int(P),
                                 "stack": int(stack), "oop": st.oop, "ip": st.ip, "target_pct": 0.02, "max_iter": 4000,
                                 "bets": "66%", "raise": "", "add_allin": 0.0, "force_allin": 0.0})
        self.assertTrue(resp["ok"], resp)
        self.assertEqual(resp["root_actions"], ["Check", "Bet(66)"])
        r0, r1 = (labels._vec(list(d), list(d.values())) for d in (st.oop, st.ip))
        root = cfr.build_round_tree(pot=P, stack=stack, bet_fracs=(0.66,), raise_fracs=(), max_raises=0)
        game = cfr.Cfr(root, r0, r1, P, cfr.Showdown(board5, P))
        for _ in range(1500):
            game.iterate()
        ev0, ev1 = game.ev_per_hand()
        for side, mine, key in (("oop", ev0, "oop"), ("ip", ev1, "ip")):
            idx = [cards.combo_index(h) for h in resp[f"{key}_hands"]]
            theirs = np.array(resp[f"{key}_ev"])
            w = np.array(resp[f"{key}_w"]) / np.sum(resp[f"{key}_w"])
            err = 100 * float((w * np.abs(mine[idx] - theirs)).sum() / P)
            self.assertLess(err, 1.0, f"{side}: range-weighted EV error {err:.2f}% of pot")
        self.assertLess(game.exploitability() / P, 0.01)


if __name__ == "__main__":
    unittest.main()
