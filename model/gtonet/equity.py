"""Exact hand-vs-range equity with card removal, per turn board (the "showdown" of architecture.md §5/§8.4).

`BoardTables(board4)` holds, for one 4-card board:
  W[h, h']  equity of h vs h' averaged over the river cards left in the deck (0.5 where undefined)
  cat6      made-hand category of each hand on the turn, outs: river cards that raise its category
Then `equity(W, r_opp)` = sum_h' D[h,h'] W[h,h'] r_opp[h'] / sum_h' D[h,h'] r_opp[h'], the conditional EV of the
doc's §4.2 (Z(h) = the opponent mass not blocked by h).  Everything is exact and differentiable-free: it feeds
both the dataset features and the equity baseline.
"""
import numpy as np
import torch

from . import cards, poker

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_C = cards.COMBO_CARDS                                                    # (1326, 2)
_INC = np.zeros((cards.NUM_COMBOS, cards.NUM_CARDS), dtype=np.float32)   # hand -> its two cards
_INC[np.arange(cards.NUM_COMBOS), _C[:, 0]] = 1
_INC[np.arange(cards.NUM_COMBOS), _C[:, 1]] = 1
_DISJOINT = ~((_C[:, None, 0] == _C[None, :, 0]) | (_C[:, None, 0] == _C[None, :, 1]) |
              (_C[:, None, 1] == _C[None, :, 0]) | (_C[:, None, 1] == _C[None, :, 1]))
_CONST = {}


def _const(device):
    if device not in _CONST:
        _CONST[device] = dict(inc=torch.tensor(_INC, device=device), disjoint=torch.tensor(_DISJOINT, device=device).float(),
                              c1=torch.tensor(_C[:, 0], device=device), c2=torch.tensor(_C[:, 1], device=device))
    return _CONST[device]


def card_masses(r: torch.Tensor) -> torch.Tensor:
    """(..., 1326) -> (..., 52): range mass on hands containing each card."""
    return r @ _const(r.device)["inc"]


def corrected_marginals(r_i: torch.Tensor, r_j: torch.Tensor) -> torch.Tensor:
    """c_i(h) = r_i(h) * Z_i(h) / sum, with Z_i(h) = mass of r_j disjoint from h (inclusion-exclusion).

    These are the marginals that make r1.v1 + r2.v2 exact (architecture.md §4.3): the solver's "normalized weights".
    """
    k = _const(r_i.device)
    m = card_masses(r_j)
    z = r_j.sum(-1, keepdim=True) - m[..., k["c1"]] - m[..., k["c2"]] + r_j
    c = r_i * z
    return c / c.sum(-1, keepdim=True).clamp_min(1e-30)


def opponent_terms(r_i: torch.Tensor, r_j: torch.Tensor, m_j: torch.Tensor = None):
    """(corrected_marginals(r_i, r_j), blocked_fraction(r_j)) from ONE card-mass computation (`m_j` = card_masses(r_j)
    when the caller already has it): the two formulas share the same gathers, and so do the two sides of a net input."""
    k = _const(r_i.device)
    m = card_masses(r_j) if m_j is None else m_j
    b = m[..., k["c1"]] + m[..., k["c2"]]
    tot = r_j.sum(-1, keepdim=True)
    c = r_i * (tot - b + r_j)
    return c / c.sum(-1, keepdim=True).clamp_min(1e-30), (b - r_j) / tot.clamp_min(1e-30)


def blocked_fraction(r_j: torch.Tensor) -> torch.Tensor:
    """(..., 1326): share of the opponent range r_j that a hand h blocks (1 - Z(h)/sum r_j)."""
    k = _const(r_j.device)
    m = card_masses(r_j)
    s = r_j.sum(-1, keepdim=True).clamp_min(1e-30)
    return (m[..., k["c1"]] + m[..., k["c2"]] - r_j) / s


class BoardTables:
    def __init__(self, board, device=DEVICE, chunk=8, keep_w=True):
        board = [cards.CARD_ID[c] if isinstance(c, str) else int(c) for c in board]
        assert len(board) == 4
        self.board, self.device = board, device
        hand_ok = ~np.isin(_C, board).any(1)                                   # (1326,)
        rivers = [c for c in range(52) if c not in board]
        hands = np.concatenate([_C, np.tile(board, (cards.NUM_COMBOS, 1))], axis=1)  # (1326, 6)
        s6 = np.where(hand_ok, poker.strength(hands), -1)
        self.cat6 = np.where(hand_ok, poker.category(s6), 0).astype(np.int8)
        rv = np.array(rivers)
        h7 = np.concatenate([np.tile(hands, (len(rivers), 1)), np.repeat(rv, cards.NUM_COMBOS)[:, None]], axis=1)
        sc = poker.strength(h7).reshape(len(rivers), cards.NUM_COMBOS)   # one vectorised call for all river cards
        ok = hand_ok[None, :] & ~(_C[None, :, :] == rv[:, None, None]).any(2)
        s7 = np.where(ok, sc, -1)
        cat7 = np.where(s7 >= 0, poker.category(s7), -1)
        self.outs = ((cat7 > self.cat6[None, :]) & (s7 >= 0)).sum(0).astype(np.int8)
        S = torch.tensor(s7, dtype=torch.float32, device=device)                 # exact: scores < 2**24
        num = torch.zeros(cards.NUM_COMBOS, cards.NUM_COMBOS, device=device)
        den = torch.zeros_like(num)
        for i in range(0, len(rivers), chunk):
            a, b = S[i:i + chunk, :, None], S[i:i + chunk, None, :]
            valid = ((a >= 0) & (b >= 0)).float()
            num += (((a > b).float() + 0.5 * (a == b).float()) * valid).sum(0)
            den += valid.sum(0)
        k = _const(device)
        self.W = torch.where(den > 0, num / den.clamp_min(1), torch.full_like(num, 0.5))
        self.WD = self.W * k["disjoint"]
        if not keep_w:
            del self.W  # the runtime resolver holds 49 boards: halve their memory
        self.hand_ok = torch.tensor(hand_ok, device=device)

    def equity(self, r_opp: torch.Tensor) -> torch.Tensor:
        """r_opp (..., 1326) raw opponent range -> (..., 1326) conditional equity of every hand vs it."""
        r = r_opp.to(self.device)
        num = r @ self.WD.T
        den = r @ _const(self.device)["disjoint"].T
        return torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.full_like(num, 0.5))

    def uniform_equity(self) -> torch.Tensor:
        return self.equity(self.hand_ok.float())


def features(board, in_oop: np.ndarray, in_ip: np.ndarray, device=DEVICE) -> dict:
    """The per-state features stored in the dataset (all range- or board-exact, nothing learned)."""
    t = BoardTables(board, device)
    r = torch.tensor(np.stack([in_oop, in_ip]), dtype=torch.float32, device=device)
    eq_oop, eq_ip = t.equity(r[1]), t.equity(r[0])   # OOP hands face the IP range, and vice versa
    return dict(f_eq_unif=t.uniform_equity().cpu().numpy(), f_eq_oop=eq_oop.cpu().numpy(), f_eq_ip=eq_ip.cpu().numpy(),
                f_cat=t.cat6, f_outs=t.outs)
