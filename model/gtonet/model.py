"""net_turn (architecture.md §4, §8): a turn-root -> (v1, v2) in R^1326, value = equity baseline + bilinear + MLP.

Targets/outputs are in normalised units  v = (EV/pot - 1/2) / (1 + SPR)  (§4.2); `to_evp` converts back.
Inputs are canonicalised (suits) by `batching`; ranges are the RAW normalised input ranges, and the card-removal
corrected marginals used by the zero-sum layer and the loss are recomputed inside from them (equity.py).
At initialisation the heads are zero, so step 0 IS the parameter-free equity baseline (§8.6).
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cards, equity

N_FEAT = 6 + 9  # eq_unif, eq_oop, eq_ip, outs, blocked_oop, blocked_ip + 9 hand categories
HAND_CARDS = torch.tensor(cards.COMBO_CARDS)  # (1326, 2)


@dataclass
class Config:
    d_card: int = 64
    d_hand: int = 128
    d_board: int = 256
    d_trunk: int = 1024
    n_blocks: int = 6
    d_ctx: int = 512
    d_head: int = 384
    n_head_layers: int = 3
    k_bilin: int = 64
    n_queries: int = 8


DOC = Config()  # ~ the 18M-parameter design of §8.4
SMALL = Config(d_card=32, d_hand=64, d_board=128, d_trunk=256, n_blocks=3, d_ctx=128, d_head=128, n_head_layers=2,
               k_bilin=32, n_queries=4)  # for the few-thousand-solve regime we are in
# Inference cost is dominated by the per-hand head (1326 hands x MLP on [E_h, ctx]); these shrink its width:
# head MACs per hand ~ SMALL 41k, S2 10k, TINY 2.6k (DOC 541k).
S2 = Config(d_card=32, d_hand=32, d_board=128, d_trunk=256, n_blocks=3, d_ctx=64, d_head=64, n_head_layers=2,
            k_bilin=16, n_queries=4)
TINY = Config(d_card=16, d_hand=16, d_board=64, d_trunk=128, n_blocks=2, d_ctx=32, d_head=32, n_head_layers=2,
              k_bilin=8, n_queries=2)
CONFIGS = {"tiny": TINY, "s2": S2, "small": SMALL, "doc": DOC}


def fourier_spr(spr: torch.Tensor) -> torch.Tensor:
    """13-dim encoding of log SPR (§8.0, operation 6): the value has kinks in SPR, a scalar captures them badly."""
    x = torch.log(spr.clamp_min(1e-3))[:, None]
    ks = (2.0 ** torch.arange(6, device=spr.device))[None, :] * math.pi
    return torch.cat([x, torch.sin(ks * x), torch.cos(ks * x)], dim=-1)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln, self.a, self.b = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return x + self.b(F.gelu(self.a(self.ln(x))))


def mlp(d_in, d_hidden, n_layers, d_out, zero_last=False):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, d_hidden), nn.GELU()]
        d = d_hidden
    last = nn.Linear(d, d_out)
    if zero_last:
        nn.init.zeros_(last.weight), nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


class NetTurn(nn.Module):
    def __init__(self, cfg: Config = SMALL):
        super().__init__()
        self.cfg, c = cfg, cfg
        self.rank, self.suit = nn.Embedding(13, c.d_card // 2), nn.Embedding(4, c.d_card // 2)   # factorised cards
        self.flop_phi = mlp(c.d_card, c.d_board, 1, c.d_board)                                    # DeepSets on the flop
        self.turn_psi = nn.Linear(c.d_card, c.d_board)                                            # the turn: its own slot
        self.board_mlp = mlp(2 * c.d_board, c.d_board, 1, c.d_board)
        self.hand_in = nn.Linear(2 * c.d_card + N_FEAT, c.d_hand)
        self.hand_board = nn.Linear(c.d_board, c.d_hand)
        self.hand_ln = nn.LayerNorm(c.d_hand)
        self.queries = nn.Parameter(0.1 * torch.randn(c.n_queries, c.d_hand))
        self.attn_key, self.attn_val = nn.Linear(c.d_hand, c.d_hand, bias=False), nn.Linear(c.d_hand, c.d_hand)
        self.attn_out = nn.Linear(c.n_queries * c.d_hand, c.d_hand)
        self.trunk_in = nn.Linear(c.d_board + 2 * 3 * c.d_hand + 13, c.d_trunk)
        self.blocks = nn.ModuleList(Block(c.d_trunk) for _ in range(c.n_blocks))
        self.ctx = nn.Sequential(nn.LayerNorm(c.d_trunk), nn.Linear(c.d_trunk, c.d_ctx))
        self.head_v = mlp(c.d_hand + c.d_ctx, c.d_head, c.n_head_layers, 2, zero_last=True)      # (v1(h), v2(h))
        self.head_pol = mlp(c.d_hand + c.d_ctx, c.d_head, 2, 3)                                   # OOP root policy slots
        self.bil_u, self.bil_v = nn.Linear(c.d_hand, 2 * c.k_bilin), nn.Linear(c.d_hand, 2 * c.k_bilin)
        nn.init.zeros_(self.bil_u.weight), nn.init.zeros_(self.bil_u.bias)                        # bilinear branch starts at 0
        self.register_buffer("hand_cards", HAND_CARDS.clone(), persistent=False)

    # -- encoders -----------------------------------------------------------------------------------------
    def card_emb(self, ids):
        return torch.cat([self.rank(ids // 4), self.suit(ids % 4)], dim=-1)

    def encode_board(self, board):
        e = self.card_emb(board)                                      # (B, 4, d_card)
        flop = self.flop_phi(e[:, :3]).sum(1)                         # order-free sum over the 3 flop cards
        return self.board_mlp(torch.cat([flop, self.turn_psi(e[:, 3])], dim=-1))

    def pool(self, E, r):
        """Range -> R in the hand space: r-weighted mean, second moment, and attention over the range (§8.2)."""
        m1 = torch.einsum("bh,bhd->bd", r, E)
        m2 = torch.einsum("bh,bhd->bd", r, E * E)
        scores = torch.einsum("bhd,qd->bqh", self.attn_key(E), self.queries) / math.sqrt(E.shape[-1])
        scores = scores + torch.log(r.clamp_min(1e-12))[:, None, :]
        scores = scores.masked_fill((r <= 0)[:, None, :], float("-inf"))
        att = torch.einsum("bqh,bhd->bqd", scores.softmax(-1), self.attn_val(E)).flatten(1)
        return torch.cat([m1, m2, self.attn_out(att)], dim=-1)

    # -- forward ------------------------------------------------------------------------------------------
    def _head_first(self, seq, E, ctx):
        """Linear([E, ctx]) = W_E E + W_c ctx + b, exactly: no (B, K, d_hand + d_ctx) concatenation is ever built."""
        lin, d = seq[0], E.shape[-1]
        h = F.linear(E, lin.weight[:, :d], lin.bias) + F.linear(ctx, lin.weight[:, d:])[:, None, :]
        for layer in list(seq)[1:]:
            h = layer(h)
        return h

    def forward(self, b: dict, policy: bool = True) -> dict:
        """Inputs are (B, 1326) vectors in ONE consistent hand order (any order: nothing here depends on the index).

        Inference options (all exact, none change the trained function):
          b["hand_cards"] (B, K, 2)  suit-relabelled card ids of the K hands: canonicalisation without permuting any
                                     vector (with `b["board"]` relabelled the same way).  Default: the fixed table.
          b["support"]   (K,) long   evaluate only these hands (e.g. the initial range's live hands); everything
                                     else is left out, and outputs have K columns in that order.
          policy=False               skip the OOP root-policy head (the flop resolver does not need it).
        """
        r_oop, r_ip, spr = b["in_oop"], b["in_ip"], b["spr"]
        c_oop, c_ip = equity.corrected_marginals(r_oop, r_ip), equity.corrected_marginals(r_ip, r_oop)   # need the full range
        bf_oop, bf_ip = equity.blocked_fraction(r_ip), equity.blocked_fraction(r_oop)
        S = b.get("support")
        pick = (lambda x: x) if S is None else (lambda x: x[:, S])  # noqa: E731
        r_oop, r_ip, c_oop, c_ip = pick(r_oop), pick(r_ip), pick(c_oop), pick(c_ip)
        f_oop, f_ip = pick(b["f_eq_oop"]), pick(b["f_eq_ip"])
        feat = torch.cat([
            torch.stack([pick(b["f_eq_unif"]), f_oop, f_ip, pick(b["f_outs"]) / 46.0, pick(bf_oop), pick(bf_ip)], dim=-1),
            F.one_hot(pick(b["f_cat"]), 9).float()], dim=-1)                              # (B, K, N_FEAT)
        bemb = self.encode_board(b["board"])
        hc = b.get("hand_cards")
        if hc is None:
            e = self.card_emb(self.hand_cards if S is None else self.hand_cards[S])       # (K, 2, d_card)
            pair = torch.cat([e[:, 0] + e[:, 1], e[:, 0] * e[:, 1]], dim=-1)[None].expand(len(spr), -1, -1)
        else:
            e = self.card_emb(hc)                                                         # (B, K, 2, d_card)
            pair = torch.cat([e[:, :, 0] + e[:, :, 1], e[:, :, 0] * e[:, :, 1]], dim=-1)  # symmetric in the two cards
        E = self.hand_ln(F.gelu(self.hand_in(torch.cat([pair, feat], -1)) + self.hand_board(bemb)[:, None, :]))
        x = torch.cat([bemb, self.pool(E, r_oop), self.pool(E, r_ip), fourier_spr(spr)], dim=-1)
        x = self.trunk_in(x)
        for blk in self.blocks:
            x = blk(x)
        ctx = self.ctx(x)

        # v = equity baseline + bilinear (linear in the opponent range) + per-hand MLP correction
        scale = (1.0 + spr)[:, None]
        base = torch.stack([(f_oop - 0.5) / scale, (f_ip - 0.5) / scale], dim=-1)
        k = self.cfg.k_bilin
        U, V = self.bil_u(E), self.bil_v(E)
        bil_oop = torch.einsum("bhk,bk->bh", U[..., :k], torch.einsum("bh,bhk->bk", r_ip, V[..., :k]))
        bil_ip = torch.einsum("bhk,bk->bh", U[..., k:], torch.einsum("bh,bhk->bk", r_oop, V[..., k:]))
        v = base + torch.stack([bil_oop, bil_ip], dim=-1) + self._head_first(self.head_v, E, ctx).float()
        v1, v2 = v[..., 0], v[..., 1]

        # hard zero-sum layer, fp32 (§4.3): s = c1.v1 + c2.v2, subtract s/2 from both
        s = (c_oop * v1).sum(-1, keepdim=True) + (c_ip * v2).sum(-1, keepdim=True)
        return dict(v_oop=v1 - s / 2, v_ip=v2 - s / 2, c_oop=c_oop, c_ip=c_ip, support=S,
                    policy=self._head_first(self.head_pol, E, ctx) if policy else None)


# -- targets, loss, conversions -------------------------------------------------------------------------------
def to_norm(evp, spr):
    """EV/pot -> normalised value (§4.2)."""
    return (evp - 0.5) / (1.0 + spr)[:, None]


def to_evp(v, spr):
    return v * (1.0 + spr)[:, None] + 0.5


def loss_fn(out: dict, b: dict, delta=0.1, eps=0.01, policy_weight=0.1):
    """L = L_value + 0.1 L_policy (§9): Huber on normalised values weighted by (eps + relative range mass)."""
    parts = {}
    total = 0.0
    for side in ("oop", "ip"):
        c, v_true = out[f"c_{side}"], to_norm(b[f"v_{side}"], b["spr"])
        present = (b[f"in_{side}"] > 0).float()                                  # only solved hands have labels
        w = present * (eps + c / c.max(-1, keepdim=True).values.clamp_min(1e-12))
        h = F.huber_loss(out[f"v_{side}"], v_true, delta=delta, reduction="none")
        total = total + ((w * h).sum(-1) / w.sum(-1).clamp_min(1e-12)).mean() / 2
    parts["value"] = float(total)
    # policy head: logits (B, 1326, 3); target strategy (B, 3, 1326); illegal slots masked out
    logits = out["policy"].masked_fill(~b["legal"][:, None, :], -1e4)
    logp = F.log_softmax(logits, dim=-1).transpose(1, 2)                             # (B, 3, 1326)
    tgt = b["strat"]
    kl = (tgt * (torch.log(tgt.clamp_min(1e-9)) - logp)).sum(1)                       # (B, 1326)
    present = (b["in_oop"] > 0).float()
    w = present * (eps + out["c_oop"] / out["c_oop"].max(-1, keepdim=True).values.clamp_min(1e-12))
    pol = ((w * kl).sum(-1) / w.sum(-1).clamp_min(1e-12)).mean()
    parts["policy"] = float(pol)
    return total + policy_weight * pol, parts


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
