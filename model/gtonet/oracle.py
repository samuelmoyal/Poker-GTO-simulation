"""Exact leaf values from the solver: validates the resolver's chance-node maths with NO network in the loop.

`ExactRiverLeaves` values the ends of a TURN round by solving every (leaf, river card) river subgame exactly with
`postflop-solver`.  A turn resolve with these leaves must reproduce the solver's own turn-root solve (same trees);
the same code path (`resolver.CachedLeaves` / `chance_average`) serves the flop with `net_turn` leaves.
"""
import numpy as np
import torch

from . import cards, labels, resolver

CPU = torch.device("cpu")


class ExactRiverLeaves(resolver.CachedLeaves):
    def __init__(self, board4, pot, stack, solver, root, floor=0.0):
        ids = [cards.CARD_ID[c] for c in board4]
        self.board4, self.pot, self.stack, self.solver, self.floor = list(board4), pot, stack, solver, floor
        self.river_cards = [c for c in range(52) if c not in ids]                                    # 48
        blocks = (torch.tensor(cards.COMBO_CARDS)[None] == torch.tensor(self.river_cards)[:, None, None]).any(-1)
        super().__init__(blocks, 44, CPU)
        self.root = [np.asarray(x, dtype=np.float64) for x in root]
        self.n_solves = 0

    def _range(self, r, keep, root):
        r = resolver.mix_with_root(torch.tensor(r)[None], torch.tensor(root)[None], torch.tensor(keep)[None], self.floor)[0].numpy()
        top = r.max()
        return {cards.combo_str(i): float(r[i] / top) for i in np.flatnonzero(r > 1e-9 * top)}

    def refresh(self, ends, sample=None):
        """sample: indices of the river cards to solve (chance sampling); None = all 48."""
        L, K = len(ends), len(self.river_cards)
        ks = range(K) if sample is None else sorted(sample)
        u = [np.zeros((L, K, cards.NUM_COMBOS), np.float32) for _ in range(2)]
        for li, (node, r0, r1) in enumerate(ends):
            x = node.contrib[0]
            p2, s2 = int(round(self.pot + 2 * x)), int(round(self.stack - x))
            for k in ks:
                c = self.river_cards[k]
                keep = (~self.blocks[k].numpy()).astype(np.float64)
                resp = self.solver.solve({"id": "o", "flop": "".join(self.board4[:3]), "turn": self.board4[3], "river": cards.DECK[c],
                                          "pot": p2, "stack": s2, "oop": self._range(r0, keep, self.root[0]),
                                          "ip": self._range(r1, keep, self.root[1]), "target_pct": 0.02, "max_iter": 3000,
                                          "bets": "66%", "raise": "", "add_allin": 0.0, "force_allin": 0.0})
                self.n_solves += 1
                assert resp["ok"], resp
                for side, uu in zip(("oop", "ip"), u):
                    uu[li, k] = labels._vec(resp[f"{side}_hands"], np.array(resp[f"{side}_ev"]) - p2 / 2)   # EV = P'/2 + u
        card_w = None
        if sample is not None:
            card_w = torch.zeros(K)
            card_w[list(ks)] = K / len(ks)
        self.store(ends, torch.tensor(u[0]), torch.tensor(u[1]), card_w)
