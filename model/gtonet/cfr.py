"""Vector DCFR on one betting round, hands as the vector axis (CLAUDE.md #13: no Python loop over combos).

Utilities are zero-sum chips relative to the round's root: each player "owns" pot/2 and pays what it invests, so
a fold by q (having put x_q in) is worth  +(pot/2 + x_q)  to the winner and  -(pot/2 + x_q)  to q, and a tie is 0.
This is the same convention on every street, so a value reached at the end of this round (`u = EV - P'/2`, where
`EV = P'/2 + u` is what the solver / net_turn report) plugs straight in as a leaf.

Counterfactual values are weighted by the opponent's reach:  cfv_p(h) = sum_{h' disjoint from h} r_opp(h') u(h, h').
The conditional EV the net predicts is  u_cond(h) = cfv_p(h) / Z_p(h)  with Z_p(h) = opponent mass not blocked by h.

Terminals are pluggable: `fold` is exact here; the end of the round is a `Terminal` (exact `Showdown` on the river,
`gtonet.resolver.NetLeaves` for a flop truncated at the turn).  A forward pass computes every node's reach first,
so all leaf evaluations can be batched (CLAUDE.md #14) before one backward pass updates regrets.
"""
from dataclasses import dataclass, field

import numpy as np

from . import cards, equity, poker

H = cards.NUM_COMBOS
_INC = equity._INC.astype(np.float64)          # (1326, 52) hand -> its two cards
_C1, _C2 = cards.COMBO_CARDS[:, 0], cards.COMBO_CARDS[:, 1]
_DISJOINT = equity._DISJOINT


def zmass(r: np.ndarray) -> np.ndarray:
    """Z(h): mass of the range r that hand h does not block (inclusion-exclusion over h's two cards)."""
    m = r @ _INC
    return r.sum() - m[_C1] - m[_C2] + r


@dataclass(eq=False)
class Node:
    kind: str                       # "action" | "fold" | "end"
    player: int = -1                # who acts (0 = OOP, 1 = IP)
    actions: list = field(default_factory=list)
    children: list = field(default_factory=list)
    folder: int = -1
    contrib: tuple = (0.0, 0.0)     # chips each player has put in this round
    idx: int = -1                   # index into the regret arrays (action nodes)
    line: str = ""                  # e.g. "x-b-c"


def build_round_tree(pot, stack, bet_fracs=(0.5,), raise_fracs=(1.0,), max_raises=1, ip_bets=True) -> Node:
    """OOP acts first.  A bet is a fraction of the pot; a raise-to is  x_opp + f * (pot + 2 x_opp)  (a pot-sized
    raise after calling).  Any action that would put the whole stack in is left out: no all-in showdown at the
    flop (documented simplification, the net was not trained at SPR 0 either)."""
    counter = [0]

    def end(x, line):
        return Node("end", contrib=tuple(x), line=line)

    def act(player, x, n_raises, facing, checked, line):
        opp = 1 - player
        node = Node("action", player=player, contrib=tuple(x), line=line)
        node.idx, counter[0] = counter[0], counter[0] + 1
        if not facing:
            node.actions.append("x")
            node.children.append(end(x, line + "x") if checked else act(opp, x, 0, False, True, line + "x-"))
            for f in (bet_fracs if (ip_bets or not checked) else ()):
                nx = list(x)
                nx[player] = x[player] + f * pot
                if nx[player] < stack:
                    node.actions.append(f"b{f:g}")
                    node.children.append(act(opp, nx, 0, True, False, line + f"b{f:g}-"))
        else:
            node.actions.append("f")
            node.children.append(Node("fold", folder=player, contrib=tuple(x), line=line + "f"))
            nx = list(x)
            nx[player] = x[opp]
            node.actions.append("c")
            node.children.append(end(nx, line + "c"))
            if n_raises < max_raises:
                for f in raise_fracs:
                    to = x[opp] + f * (pot + 2 * x[opp])
                    if to < stack:
                        nx = list(x)
                        nx[player] = to
                        node.actions.append(f"r{f:g}")
                        node.children.append(act(opp, nx, n_raises + 1, True, False, line + f"r{f:g}-"))
        return node

    return act(0, [0.0, 0.0], 0, False, False, "")


def walk(node):
    yield node
    for c in node.children:
        yield from walk(c)


class Showdown:
    """Exact river showdown with card removal: cfv_p(h) = (pot/2 + x) * sum_h' r_opp(h') sign(s_h - s_h')."""

    def __init__(self, board5, pot):
        ids = np.array([cards.CARD_ID[c] for c in board5])
        hands = np.concatenate([cards.COMBO_CARDS, np.tile(ids, (H, 1))], axis=1)
        ok = ~np.isin(cards.COMBO_CARDS, ids).any(1)
        s = np.where(ok, poker.strength(hands), -1)
        valid = _DISJOINT & ok[:, None] & ok[None, :]
        self.M = np.where(valid, np.sign(s[:, None] - s[None, :]), 0).astype(np.float64)
        self.pot = pot

    def cfv(self, node, r0, r1):
        u = self.pot / 2 + node.contrib[0]         # contributions are equal at a showdown
        return u * (self.M @ r1), u * (self.M @ r0)   # OOP's hands face r1 and vice versa


class Cfr:
    """Simultaneous vector DCFR (alpha, beta, gamma as in architecture.md §3.4: 1.5, 0, 2)."""

    def __init__(self, root, r0, r1, pot, terminal, alpha=1.5, beta=0.0, gamma=2.0):
        self.root, self.pot, self.terminal = root, pot, terminal
        self.r0, self.r1 = np.asarray(r0, np.float64), np.asarray(r1, np.float64)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.actions = [n for n in walk(root) if n.kind == "action"]
        self.regret = [np.zeros((len(n.actions), H)) for n in self.actions]
        self.ssum = [np.zeros((len(n.actions), H)) for n in self.actions]
        self.t = 0

    def sigma(self, node) -> np.ndarray:
        pos = np.maximum(self.regret[node.idx], 0.0)
        tot = pos.sum(0)
        return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / len(node.actions))

    def avg_sigma(self, node) -> np.ndarray:
        s = self.ssum[node.idx]
        tot = s.sum(0)
        return np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), 1.0 / len(node.actions))

    def _fold(self, node, r0, r1):
        f, w = node.folder, 1 - node.folder
        u = self.pot / 2 + node.contrib[f]
        r = (r0, r1)
        cfv = [None, None]
        cfv[f], cfv[w] = -u * zmass(r[w]), u * zmass(r[f])
        return cfv[0], cfv[1]

    def _terminal(self, node, r0, r1):
        return self._fold(node, r0, r1) if node.kind == "fold" else self.terminal.cfv(node, r0, r1)

    def end_reaches(self, strategy=None):
        """Forward pass: [(end node, r0, r1)] under the current strategy (the batch a leaf evaluator wants)."""
        out = []

        def fwd(node, r0, r1):
            if node.kind == "end":
                out.append((node, r0, r1))
            elif node.kind == "action":
                s = strategy(node) if strategy else self.sigma(node)
                for a, ch in enumerate(node.children):
                    fwd(ch, r0 * s[a] if node.player == 0 else r0, r1 * s[a] if node.player == 1 else r1)

        fwd(self.root, self.r0, self.r1)
        return out

    def iterate(self):
        """One DCFR iteration: regrets, discounted averages.  Returns nothing; read `avg_sigma`."""
        self.t += 1
        t = self.t
        pw = t ** self.alpha / (t ** self.alpha + 1)
        nw = t ** self.beta / (t ** self.beta + 1)
        gw = (t / (t + 1)) ** self.gamma

        def go(node, r0, r1):
            if node.kind != "action":
                return self._terminal(node, r0, r1)
            p, s = node.player, self.sigma(node)
            ch = [go(c, r0 * s[a] if p == 0 else r0, r1 * s[a] if p == 1 else r1) for a, c in enumerate(node.children)]
            mine = np.stack([c[p] for c in ch])                       # (A, H) my cfv per action
            cfv_p = (s * mine).sum(0)
            cfv_o = sum(c[1 - p] for c in ch)
            R = self.regret[node.idx]
            R[:] = np.where(R > 0, R * pw, R * nw) + (mine - cfv_p)
            self.ssum[node.idx] += gw * (r0 if p == 0 else r1) * s
            return (cfv_p, cfv_o) if p == 0 else (cfv_o, cfv_p)

        go(self.root, self.r0, self.r1)

    # -- evaluation of the AVERAGE strategy (values, exploitability); exact terminals only ------------------
    def profile_cfv(self):
        """cfv of both players when both play the average strategy."""
        def go(node, r0, r1):
            if node.kind != "action":
                return self._terminal(node, r0, r1)
            p, s = node.player, self.avg_sigma(node)
            ch = [go(c, r0 * s[a] if p == 0 else r0, r1 * s[a] if p == 1 else r1) for a, c in enumerate(node.children)]
            cfv_p = sum(s[a] * ch[a][p] for a in range(len(ch)))
            cfv_o = sum(c[1 - p] for c in ch)
            return (cfv_p, cfv_o) if p == 0 else (cfv_o, cfv_p)

        return go(self.root, self.r0, self.r1)

    def best_response(self, br: int) -> np.ndarray:
        """cfv of player `br` playing a best response to the opponent's average strategy."""
        def go(node, r0, r1):
            if node.kind != "action":
                return self._terminal(node, r0, r1)
            p = node.player
            s = self.avg_sigma(node) if p != br else None
            ch = [go(c, r0 * s[a] if p == 0 and s is not None else r0, r1 * s[a] if p == 1 and s is not None else r1)
                  for a, c in enumerate(node.children)]
            if p == br:
                mine = np.stack([c[br] for c in ch]).max(0)
                other = sum(c[1 - br] for c in ch)   # unused by the caller when br acts (its value is `mine`)
                return (mine, other) if br == 0 else (other, mine)
            cfv_br = sum(c[br] for c in ch)
            other = sum(s[a] * ch[a][1 - br] for a in range(len(ch)))
            return (cfv_br, other) if br == 0 else (other, cfv_br)

        return go(self.root, self.r0, self.r1)[br]

    def norm(self) -> float:
        """N = sum_h r0(h) Z_0(h): total weight of compatible hand pairs (turns a cfv sum into an expectation)."""
        return float(self.r0 @ zmass(self.r1))

    def exploitability(self) -> float:
        """(BR_0 + BR_1) / 2 in chips per compatible pair, against the average strategy (exact terminals only)."""
        return 0.5 * (float(self.r0 @ self.best_response(0)) + float(self.r1 @ self.best_response(1))) / self.norm()

    def ev_per_hand(self):
        """(EV_oop, EV_ip) in chips per hand, conditional on holding it, average strategy: pot/2 + cfv / Z."""
        c0, c1 = self.profile_cfv()
        z0, z1 = zmass(self.r1), zmass(self.r0)
        return (self.pot / 2 + np.where(z0 > 1e-12, c0 / np.where(z0 > 1e-12, z0, 1), 0.0),
                self.pot / 2 + np.where(z1 > 1e-12, c1 / np.where(z1 > 1e-12, z1, 1), 0.0))
