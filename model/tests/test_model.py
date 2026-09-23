import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import batching, cards, dataset, equity, evaluate, labels, model as M, paths, sensitivity  # noqa: E402

try:
    D = dataset.load()
except FileNotFoundError:
    D = None
DEV = torch.device("cpu")  # tests are exact-arithmetic checks, keep them device-independent


@unittest.skipUnless(D, "needs featurised shards")
class PermutationTest(unittest.TestCase):
    """CLAUDE.md: permuted board + reindexed vectors must give the same thing; a slip here has no visible symptom."""

    def test_roundtrip(self):
        x = torch.rand(4, 3, 1326)
        p = torch.tensor([0, 5, 17, 23])
        torch.testing.assert_close(batching.unapply_perm(batching.apply_perm(x, p), p), x)

    def test_features_are_equivariant(self):
        rng = np.random.default_rng(0)
        for i in rng.choice(len(D["id"]), 3, replace=False):
            p = int(rng.integers(1, 24))
            pt = torch.tensor([p])
            board = [int(c) for c in D["board"][i]]
            new_board = [int(batching.CARD_PERMS[p][c]) for c in board]
            a = batching.apply_perm(torch.tensor(np.array(D["in_oop"][i]))[None].float(), pt)[0].numpy()
            b = batching.apply_perm(torch.tensor(np.array(D["in_ip"][i]))[None].float(), pt)[0].numpy()
            f = equity.features(new_board, a, b, DEV)
            for key in ("f_eq_unif", "f_eq_oop", "f_eq_ip", "f_cat", "f_outs"):
                want = batching.apply_perm(torch.tensor(np.array(D[key][i]))[None].float(), pt)[0].numpy()
                np.testing.assert_allclose(f[key], want, atol=1e-5, err_msg=key)

    def test_canonical_form_is_permutation_invariant(self):
        data = batching.TurnData({k: v for k, v in D.items()}, DEV)
        rng = np.random.default_rng(1)
        checked = 0
        for i in rng.choice(data.n, 30, replace=False):
            b0 = data.batch([i], "canonical")
            p = int(rng.integers(0, 24))
            # the same state, suits relabelled by p (board and every per-hand vector), then canonicalised again
            pt = torch.tensor([p])
            moved = {k: batching.apply_perm(v[i][None], pt) for k, v in data.hand.items()}
            board = batching.CARD_PERMS[p][data.board[i]][None]
            canon = torch.tensor(batching.canonical_perm_ids(board.numpy()))
            flop = sorted(int(c) for c in board[0][:3])
            ties = sum(1 for perm in cards.SUIT_PERMS
                       if (tuple(sorted(4 * (c // 4) + perm[c % 4] for c in flop)), 4 * (int(board[0][3]) // 4) + perm[int(board[0][3]) % 4])
                       == min((tuple(sorted(4 * (c // 4) + q[c % 4] for c in flop)), 4 * (int(board[0][3]) // 4) + q[int(board[0][3]) % 4]) for q in cards.SUIT_PERMS))
            if ties > 1:
                continue  # board with a symmetry: canonical form is unique only up to it
            b1 = {k: batching.apply_perm(v, canon) for k, v in moved.items()}
            for k in ("in_oop", "in_ip", "f_eq_oop"):
                torch.testing.assert_close(b1[k], b0[k], atol=1e-6, rtol=0, msg=k)
            checked += 1
        self.assertGreater(checked, 5)


@unittest.skipUnless(D, "needs featurised shards")
class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = batching.TurnData(D, DEV)
        cls.b = cls.data.batch(list(range(8)), "canonical")

    def test_zero_sum_layer(self):
        net = M.NetTurn(M.SMALL)
        for p in net.parameters():  # a random (non-zero) net must still be exactly zero-sum
            torch.nn.init.normal_(p, 0, 0.05) if p.dim() > 1 else None
        o = net(self.b)
        s = (o["c_oop"] * o["v_oop"]).sum(-1) + (o["c_ip"] * o["v_ip"]).sum(-1)
        self.assertLess(float(s.abs().max()), 1e-5)

    def test_step_zero_is_the_equity_baseline(self):
        net = M.NetTurn(M.SMALL)
        o = net(self.b)
        want = M.to_norm(self.b["f_eq_oop"], self.b["spr"])
        torch.testing.assert_close(o["v_oop"], want, atol=1e-5, rtol=0)
        torch.testing.assert_close(M.to_evp(o["v_ip"], self.b["spr"]), self.b["f_eq_ip"], atol=1e-5, rtol=0)

    def test_targets_are_zero_sum_in_normalised_units(self):
        c_oop = equity.corrected_marginals(self.b["in_oop"], self.b["in_ip"])
        c_ip = equity.corrected_marginals(self.b["in_ip"], self.b["in_oop"])
        s = (c_oop * M.to_norm(self.b["v_oop"], self.b["spr"])).sum(-1) + (c_ip * M.to_norm(self.b["v_ip"], self.b["spr"])).sum(-1)
        self.assertLess(float(s.abs().max()), 1e-4)  # r.v = 0 after the -1/2 shift (labels are float32 chips/pot)

    def test_loss_backward(self):
        net = M.NetTurn(M.SMALL)
        loss, parts = M.loss_fn(net(self.b), self.b)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def _random_net(self):
        torch.manual_seed(3)
        net = M.NetTurn(M.SMALL).eval()
        for p in net.parameters():
            torch.nn.init.normal_(p, 0, 0.05) if p.dim() > 1 else None      # non-zero heads: a real test
        return net

    def test_relabelling_cards_equals_permuting_vectors(self):
        """Canonicalisation for inference: rename suits on the board and on each hand's two cards, leave every vector
        in its original order.  The net is a set function of hands, so this must equal permute -> forward -> unpermute."""
        net = self._random_net()
        idx = list(range(8))
        canon, orig = self.data.batch(idx, "canonical"), self.data.batch(idx, "identity")
        with torch.no_grad():
            ref = net(canon)
            ref_oop = batching.unapply_perm(ref["v_oop"], canon["perm"])
            ref_ip = batching.unapply_perm(ref["v_ip"], canon["perm"])
            relabel = dict(orig, board=canon["board"],
                           hand_cards=batching.CARD_PERMS[canon["perm"]][:, torch.tensor(cards.COMBO_CARDS)])   # (B, 1326, 2)
            new = net(relabel)
        self.assertLess(float((new["v_oop"] - ref_oop).abs().max()), 1e-5)
        self.assertLess(float((new["v_ip"] - ref_ip).abs().max()), 1e-5)

    def test_support_only_matches_full_on_those_hands(self):
        net = self._random_net()
        orig = self.data.batch(list(range(8)), "identity")
        live = ((orig["in_oop"] > 0) | (orig["in_ip"] > 0)).any(0)
        support = torch.nonzero(live).squeeze(1)
        self.assertLess(len(support), cards.NUM_COMBOS)                       # the test is only meaningful if it drops hands
        with torch.no_grad():
            full, sub = net(orig), net(dict(orig, support=support), policy=False)
        self.assertIsNone(sub["policy"])
        self.assertLess(float((sub["v_oop"] - full["v_oop"][:, support]).abs().max()), 1e-5)
        self.assertLess(float((sub["v_ip"] - full["v_ip"][:, support]).abs().max()), 1e-5)
        s = (sub["c_oop"] * sub["v_oop"]).sum(-1) + (sub["c_ip"] * sub["v_ip"]).sum(-1)
        self.assertLess(float(s.abs().max()), 1e-5)                           # zero-sum still exact on the support

    def test_equity_baseline_reacts_to_ranges(self):
        net = M.NetTurn(M.SMALL)  # at init == equity baseline, which must move when the opponent's range does
        shifts = [sensitivity.range_sensitivity(net, D, i, DEV)["mean_abs_shift_pct"] for i in range(4)]
        self.assertGreater(min(shifts), 1.0)  # > 1% of pot


if __name__ == "__main__":
    unittest.main()
