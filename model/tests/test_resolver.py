import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import batching, cards, cfr, flopdump, model as M, resolver  # noqa: E402


class ChanceAverageTest(unittest.TestCase):
    """The leaf aggregation, against a brute-force enumeration (no solver, no net): the 'leaf' is a round where both
    players just check down, so its exact cfv at every river card is a plain showdown."""

    def test_matches_brute_force(self):
        board4 = ["Qs", "Th", "5d", "5h"]
        ids = [cards.CARD_ID[c] for c in board4]
        rng = np.random.default_rng(0)
        ok = ~np.isin(cards.COMBO_CARDS, ids).any(1)
        r0, r1 = rng.random(cfr.H) * ok * (rng.random(cfr.H) < 0.6), rng.random(cfr.H) * ok * (rng.random(cfr.H) < 0.6)
        pot = 100.0
        rivers = [c for c in range(52) if c not in ids]
        blocks = torch.tensor((cards.COMBO_CARDS[None] == np.array(rivers)[:, None, None]).any(-1))
        u0 = np.zeros((1, len(rivers), cfr.H))
        u1 = np.zeros_like(u0)
        brute0, brute1 = np.zeros(cfr.H), np.zeros(cfr.H)
        for k, c in enumerate(rivers):
            sd = cfr.Showdown(board4 + [cards.DECK[c]], pot)                     # M[h,h'] = sign, 0 if blocked
            keep = (~blocks[k].numpy()).astype(np.float64)
            r0c, r1c = r0 * keep, r1 * keep
            z0, z1 = cfr.zmass(r1c), cfr.zmass(r0c)
            c0, c1 = sd.cfv(type("N", (), {"contrib": (0.0, 0.0)}), r0c, r1c)   # exact showdown cfv on river card c
            u0[0, k] = np.where(z0 > 1e-12, c0 / np.where(z0 > 1e-12, z0, 1), 0)   # conditional EV, what a net/solver returns
            u1[0, k] = np.where(z1 > 1e-12, c1 / np.where(z1 > 1e-12, z1, 1), 0)
            brute0 += c0 / 44                                                    # (1/44) sum_c cfv_c: the definition
            brute1 += c1 / 44
        cpu = torch.device("cpu")
        ub0, ub1 = resolver.chance_average(torch.tensor(u0, dtype=torch.float32), torch.tensor(u1, dtype=torch.float32),
                                           torch.tensor(r0[None], dtype=torch.float32), torch.tensor(r1[None], dtype=torch.float32),
                                           blocks, 44, cpu)
        got0, got1 = cfr.zmass(r1) * ub0[0], cfr.zmass(r0) * ub1[0]
        scale = max(np.abs(brute0).max(), np.abs(brute1).max())
        self.assertLess(np.abs(got0 - brute0).max() / scale, 1e-4)
        self.assertLess(np.abs(got1 - brute1).max() / scale, 1e-4)


class NetLeavesTest(unittest.TestCase):
    """The optimised leaf evaluation (relabelled cards, initial-range hands only, no policy head, one pass) against a
    plain per-state reference: permute the vectors, run the FULL net, un-permute."""

    def test_matches_per_state_reference(self):
        for fast in (False, True):
            with self.subTest(fast=fast):
                self._check(fast)

    def _check(self, fast):
        from gtonet import equity, labels
        flop = ["Ts", "7s", "3h"]
        ip, oop = flopdump.root_reach("BTN_vs_BB", flop)
        r0, r1 = (labels._vec(list(d), list(d.values())) for d in (oop, ip))
        torch.manual_seed(1)
        net = M.NetTurn(M.TINY).eval()
        for p in net.parameters():
            torch.nn.init.normal_(p, 0, 0.05) if p.dim() > 1 else None
        cpu = torch.device("cpu")
        turn_cards = [cards.CARD_ID[c] for c in ("2c", "9d", "Kh", "Ac")]
        leaves = resolver.NetLeaves(net, flop, 55.0, 975.0, cpu, chunk=3, root=(r0, r1), turn_cards=turn_cards, fast=fast)
        ref_tables = [equity.BoardTables(flop + [cards.DECK[c]], cpu) for c in turn_cards]   # independent of the code under test
        self.assertLess(len(leaves.support), cards.NUM_COMBOS)
        rng = np.random.default_rng(2)
        ends = [(type("N", (), {"contrib": (x, x)})(), r0 * rng.random(cfr.H), r1 * rng.random(cfr.H)) for x in (0.0, 27.5)]
        leaves.refresh(ends)

        ref = resolver.CachedLeaves(leaves.blocks, resolver.CHANCE, cpu)
        K = len(turn_cards)
        u = [torch.zeros(len(ends), K, cfr.H) for _ in range(2)]
        keep = (~leaves.blocks).float()
        for i, (node, e0, e1) in enumerate(ends):
            pot2, stack2 = 55.0 + 2 * node.contrib[0], 975.0 - node.contrib[0]
            for j in range(K):
                n0 = resolver.mix_with_root(torch.tensor(e0, dtype=torch.float32)[None], leaves.root[0][0], keep[j][None], leaves.floor)
                n1 = resolver.mix_with_root(torch.tensor(e1, dtype=torch.float32)[None], leaves.root[1][0], keep[j][None], leaves.floor)
                p = leaves.canon[j][None]
                one = dict(in_oop=n0, in_ip=n1, f_eq_oop=ref_tables[j].equity(n1), f_eq_ip=ref_tables[j].equity(n0),
                           f_eq_unif=ref_tables[j].uniform_equity()[None], f_cat=torch.tensor(ref_tables[j].cat6, dtype=torch.long)[None],
                           f_outs=torch.tensor(ref_tables[j].outs, dtype=torch.float32)[None],
                           spr=torch.tensor([stack2 / pot2]))
                b = {k: batching.apply_perm(v, p) for k, v in one.items() if k != "spr"}
                b["spr"], b["board"] = one["spr"], torch.gather(batching.CARD_PERMS[p], 1, leaves.boards[j][None])
                with torch.no_grad():
                    o = net(b)                                                                         # full hands, with policy head
                for uu, key in zip(u, ("v_oop", "v_ip")):
                    uu[i, j] = (pot2 + stack2) * batching.unapply_perm(o[key], p)[0]
        ref.store(ends, u[0], u[1])
        live = torch.zeros(cfr.H, dtype=torch.bool)
        live[leaves.support] = True
        for node, _, _ in ends:
            for side in (0, 1):
                got, want = leaves.ubar[id(node)][side], ref.ubar[id(node)][side]
                scale = np.abs(want[live.numpy()]).max()
                self.assertLess(np.abs(got - want)[live.numpy()].max() / scale, 1e-4)   # identical on the initial range's hands



class NetFreeLeavesTest(unittest.TestCase):
    """`refresh(net_free=True)` is what the net says before any learning: with its heads at their zero initialisation the
    net's output IS the equity baseline, so both paths must give the same leaf values (on the hands the net evaluates)."""

    def test_equals_a_zero_initialised_net(self):
        from gtonet import labels
        flop = ["Ts", "7s", "3h"]
        ip, oop = flopdump.root_reach("BTN_vs_BB", flop)
        r0, r1 = (labels._vec(list(d), list(d.values())) for d in (oop, ip))
        cpu = torch.device("cpu")
        turn_cards = [cards.CARD_ID[c] for c in ("2c", "9d", "Kh", "Ac")]
        net = M.NetTurn(M.TINY).eval()
        leaves = resolver.NetLeaves(net, flop, 55.0, 975.0, cpu, root=(r0, r1), turn_cards=turn_cards)
        rng = np.random.default_rng(3)
        ends = [(type("N", (), {"contrib": (x, x)})(), r0 * rng.random(cfr.H), r1 * rng.random(cfr.H)) for x in (0.0, 27.5)]
        leaves.refresh(ends)
        with_net = {id(n): tuple(u.copy() for u in leaves.ubar[id(n)]) for n, _, _ in ends}
        leaves.refresh(ends, net_free=True)
        live = np.zeros(cfr.H, dtype=bool)
        live[leaves.support.numpy()] = True
        for node, _, _ in ends:
            for side in (0, 1):
                np.testing.assert_allclose(leaves.ubar[id(node)][side][live], with_net[id(node)][side][live], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
