"""Torch dataset: suit canonicalisation / augmentation, hand-index permutations and root-strategy slots.

A suit permutation relabels the board cards AND reindexes every per-hand vector (`cards.combo_perm`); doing only
one of them silently trains on noise, so both go through `apply_perm`, and tests check the round trip.
`canonical_perm_ids` picks, per board, the permutation with the smallest sorted (flop, turn) key: 24 equivalent
inputs become one (architecture.md §8.0, operation 7).
"""
import numpy as np
import torch

from . import cards

CARD_PERMS = torch.tensor(np.stack([cards.card_perm(p) for p in cards.SUIT_PERMS]))     # (24, 52)
COMBO_PERMS = torch.tensor(np.stack([cards.combo_perm(p) for p in cards.SUIT_PERMS]))   # (24, 1326)
SLOT = {"Check": 0, "Bet": 1, "AllIn": 2}   # root actions -> fixed policy slots
N_SLOTS = 3
PER_HAND = ("in_oop", "in_ip", "v_oop", "v_ip", "f_eq_unif", "f_eq_oop", "f_eq_ip", "f_cat", "f_outs")


def canonical_perm_ids(boards) -> np.ndarray:
    out = np.zeros(len(boards), dtype=np.int64)
    for n, b in enumerate(boards):
        best = None
        for pi, perm in enumerate(cards.SUIT_PERMS):
            new = [4 * (int(c) // 4) + perm[int(c) % 4] for c in b]
            key = (tuple(sorted(new[:3])), new[3])
            if best is None or key < best:
                best, out[n] = key, pi
    return out


def slot_strategy(root_actions, root_strategy):
    """[[Check, Bet(79), AllIn(940)]-style list, (4*1326,)] -> ((3, 1326) strategy in fixed slots, (3,) legal mask)."""
    strat = np.zeros((N_SLOTS, cards.NUM_COMBOS), dtype=np.float32)
    legal = np.zeros(N_SLOTS, dtype=bool)
    rs = np.asarray(root_strategy).reshape(4, -1)
    for i, a in enumerate(root_actions):
        s = SLOT[a.split("(")[0]]
        strat[s], legal[s] = rs[i], True
    return strat, legal


def apply_perm(x: torch.Tensor, perm_ids: torch.Tensor) -> torch.Tensor:
    """Relabel the last (hand) axis: new[b, ..., cperm[b, k]] = old[b, ..., k]."""
    cp = COMBO_PERMS.to(x.device)[perm_ids]                       # (B, 1326)
    if x.dim() == 3:
        cp = cp[:, None, :].expand_as(x)
    return torch.zeros_like(x).scatter(-1, cp, x)


def unapply_perm(x: torch.Tensor, perm_ids: torch.Tensor) -> torch.Tensor:
    cp = COMBO_PERMS.to(x.device)[perm_ids]
    if x.dim() == 3:
        cp = cp[:, None, :].expand_as(x)
    return torch.gather(x, -1, cp)


class TurnData:
    """A featurised split held on `device`; `batch(idx, mode)` returns model-ready tensors."""

    def __init__(self, d: dict, device):
        self.device, self.n = device, len(d["id"])
        t = lambda k, dt=torch.float32: torch.tensor(np.asarray(d[k]), dtype=dt, device=device)  # noqa: E731
        self.hand = {k: t(k, torch.long if k == "f_cat" else torch.float32) for k in PER_HAND}
        self.board = t("board", torch.long)
        self.spr = t("spr")
        strat, legal = zip(*(slot_strategy(a, s) for a, s in zip(d["root_actions"], d["root_strategy"])))
        self.strat = torch.tensor(np.stack(strat), device=device)
        self.legal = torch.tensor(np.stack(legal), device=device)
        self.canon = torch.tensor(canonical_perm_ids(d["board"]), device=device)
        self.meta = {k: d[k] for k in ("id", "kind", "perturbed", "split", "flop_key")}

    def batch(self, idx, mode="canonical") -> dict:
        idx = torch.as_tensor(idx, device=self.device)
        if mode == "canonical":
            p = self.canon[idx]
        elif mode == "augment":
            p = torch.randint(0, 24, (len(idx),), device=self.device)
        else:
            p = torch.zeros(len(idx), dtype=torch.long, device=self.device)
        b = {k: apply_perm(v[idx], p) for k, v in self.hand.items()}
        b["board"] = torch.gather(CARD_PERMS.to(self.device)[p], 1, self.board[idx])
        b["spr"], b["perm"] = self.spr[idx], p
        b["strat"], b["legal"] = apply_perm(self.strat[idx], p), self.legal[idx]
        return b
