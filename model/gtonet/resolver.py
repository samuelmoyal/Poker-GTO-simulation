"""Real-time flop resolving (architecture.md §12): a truncated DCFR over the flop betting tree whose leaves are
turn-roots valued by `net_turn`.

Leaf value of a line that reaches the turn (both players in for x, pot' = pot + 2x, stack' = stack - x):
    cfv_p(h) = (1/45) sum_{c not in h} Z_c(h) * u_c(h),     u = (pot' + stack') * v    (net output v, chips)
where c runs over the 49 possible turn cards, Z_c(h) is the opponent mass left after removing card c and h's cards
(45 = 52 - 3 flop - 4 hole cards: the chance weight of a card given both hands), and u_c(h) is net_turn's
conditional EV for the board flop+c and the ranges reached at the leaf with card c removed.

Between refreshes (every `refresh` iterations, all leaves x 49 cards in ONE batched pass, CLAUDE.md #14) the net
value is frozen as  ubar(h) = cfv(h) / Z(h)  and the cfv is rescaled by the CURRENT opponent mass:  cfv = Z_now * ubar.
"""
import time

import numpy as np
import torch

from . import batching, cards, cfr, equity, fastboard, model as M

CHANCE = 45  # flop -> turn: 52 - 3 flop cards - 4 hole cards.  (turn -> river: 44)


def zmass_t(r, device):
    k = equity._const(device)
    m = r @ k["inc"]
    return r.sum(-1, keepdim=True) - m[..., k["c1"]] - m[..., k["c2"]] + r


def chance_average(u_oop, u_ip, R0, R1, blocks, chance, device, card_w=None):
    """Frozen leaf values from per-card conditional EVs.

    u_*: (L, K, 1326) chips, conditional EV of each hand for each of the K next cards; R0/R1: (L, 1326) raw reach at the
    leaf; blocks: (K, 1326) bool, hands containing card k.  Returns ubar (numpy float64, (L, 1326) x 2) such that
    cfv_p(h) = Z_p(h) * ubar_p(h) with Z the CURRENT opponent mass not blocked by h.
    """
    keep = (~blocks).float()
    wk = keep if card_w is None else keep * card_w.to(device)[:, None]
    r0c, r1c = R0[:, None, :] * keep[None], R1[:, None, :] * keep[None]
    out = []
    for u, r_opp_c, r_opp in ((u_oop, r1c, R1), (u_ip, r0c, R0)):
        z_c, z = zmass_t(r_opp_c, device), zmass_t(r_opp, device)
        ub = (wk[None] * z_c * u).sum(1) / chance / z.clamp_min(1e-12)
        out.append(torch.where(z > 1e-12, ub, torch.zeros_like(ub)).cpu().numpy().astype(np.float64))
    return out


def mix_with_root(r, root, keep, floor):
    """The eps-floor of architecture.md §9, as a mixture with the INITIAL range: input = (1-m) r/|r| + m root/|root|.

    CFR needs each hand's value at a leaf even where its reach is 0 (that is what lets an unplayed action become
    attractive again) and a leaf a player never reaches has an empty range; a net (or solver) trained/asked only on
    in-range hands says nothing about those.  m = floor normally, 1 when the leaf is unreachable.  r, root: (..., 1326)."""
    r, rt = r * keep, root * keep
    tot = r.sum(-1, keepdim=True)
    rt = rt / rt.sum(-1, keepdim=True).clamp_min(1e-30)
    empty = tot <= 1e-9 * root.sum(-1, keepdim=True)
    m = torch.where(empty, torch.ones_like(tot), torch.full_like(tot, float(floor)))
    return (1 - m) * r / tot.clamp_min(1e-30) + m * rt


class CachedLeaves:
    """Terminal for `cfr.Cfr`: leaf cfv = Z_now * ubar, ubar refreshed by `store` (called by a subclass' `refresh`)."""

    def __init__(self, blocks, chance, device):
        self.blocks, self.chance, self.device = blocks, chance, device
        self.ubar = {}

    def store(self, ends, u_oop, u_ip, card_w=None, R=None, blocks=None):
        """u_*: (L, K', 1326) values for K' next cards; `blocks` (K', 1326) and `card_w` (K',) describe those cards (default:
        all of them); `R` = the (R0, R1) tensors when the caller has them already."""
        if R is None:
            R = [torch.tensor(np.stack([e[i] for e in ends]), dtype=torch.float32, device=self.device) for i in (1, 2)]
        ub0, ub1 = chance_average(u_oop, u_ip, R[0], R[1], self.blocks if blocks is None else blocks, self.chance, self.device, card_w)
        for i, e in enumerate(ends):
            self.ubar[id(e[0])] = (ub0[i], ub1[i])

    def cfv(self, node, r0, r1):
        u0, u1 = self.ubar[id(node)]
        return cfr.zmass(r1) * u0, cfr.zmass(r0) * u1


class NetLeaves(CachedLeaves):
    """Leaves valued by `net_turn`, all (leaf x turn card) states of a refresh in one batched pass.

    Inference shortcuts, all exact (tests/test_model.py, tests/test_resolver.py):
      * suits are canonicalised by renaming the cards of the board and of each hand once per turn card at construction,
        never by permuting a vector (the net is a set function of hands);
      * only the hands of the INITIAL range (union of both players') are evaluated: CFR needs values at leaves for hands
        with zero reach, but never for hands outside the initial range;
      * the OOP root-policy head is skipped."""

    def __init__(self, net, flop, pot, stack, device, chunk=36, root=None, floor=0.003, turn_cards=None, use_support=True,
                 fast=None):
        self.net, self.device, self.chunk, self.pot, self.stack = net, device, chunk, pot, stack
        self.root = None if root is None else [torch.tensor(np.asarray(x), dtype=torch.float32, device=device)[None, None, :] for x in root]
        self.floor = floor
        self.flop = [cards.CARD_ID[c] for c in flop]
        self.turn_cards = [c for c in range(52) if c not in self.flop] if turn_cards is None else list(turn_cards)   # 49
        self.fast = fastboard.available() if fast is None else fast
        t0 = time.time()
        if self.fast:        # tools/boardlib (Rust): boards built in parallel, equity by sorted prefix sums
            self.fb, self.tables = fastboard.FastBoards([self.flop + [c] for c in self.turn_cards]), None
            self.wd = torch.from_numpy(self.fb.win_tables()).to(device)                        # (K, 1326, 1326), built in Rust
            self.disjoint = equity._const(device)["disjoint"]
            self.f_unif = torch.from_numpy(self.fb.uniform_equity()).to(device)                # (K, 1326)
            self.f_cat = torch.from_numpy(self.fb.cat6.astype(np.int64)).to(device)
            self.f_outs = torch.from_numpy(self.fb.outs.astype(np.float32)).to(device)
        else:                # numpy evaluator + a 1326x1326 win table per board on `device`
            self.tables = [equity.BoardTables(self.flop + [c], device, keep_w=False) for c in self.turn_cards]
            st = lambda f: torch.stack([f(t) for t in self.tables])  # noqa: E731
            self.f_unif = st(lambda t: t.uniform_equity())
            self.f_cat = torch.tensor(np.stack([t.cat6 for t in self.tables]), dtype=torch.long, device=device)
            self.f_outs = torch.tensor(np.stack([t.outs for t in self.tables]), dtype=torch.float32, device=device)
        self.t_tables = time.time() - t0
        self.boards = torch.tensor([self.flop + [c] for c in self.turn_cards], device=device)
        self.canon = torch.tensor(batching.canonical_perm_ids(self.boards.cpu().numpy()), device=device)
        relabel = batching.CARD_PERMS.to(device)[self.canon]                              # (K, 52): card -> canonical card
        self.boards_c = torch.gather(relabel, 1, self.boards)                             # (K, 4)
        self.support = None
        combos = torch.tensor(cards.COMBO_CARDS, device=device)                           # (1326, 2)
        if root is not None and use_support:
            live = (np.asarray(root[0]) > 0) | (np.asarray(root[1]) > 0)
            self.support = torch.tensor(np.flatnonzero(live), device=device)
            combos = combos[self.support]
        self.hand_cards_c = relabel[:, combos]                                            # (K, K_hands, 2) canonical card ids
        cc = torch.tensor(self.turn_cards, device=device)
        blocks = (torch.tensor(cards.COMBO_CARDS, device=device)[None, :, :] == cc[:, None, None]).any(-1)  # (K,1326)
        super().__init__(blocks, CHANCE, device)
        self.timing = dict(features=0.0, net=0.0, post=0.0, refreshes=0)

    def _equity(self, k, r):
        """Conditional equity of every hand vs the range r (L, 1326) on turn board k."""
        if not self.fast:
            return self.tables[k].equity(r)
        num, den = r @ self.wd[k].T, r @ self.disjoint.T
        return torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.full_like(num, 0.5))

    def _equity_all(self, ks, r):
        """`_equity` for every (range, card) pair at once.  r: (M, Ks, 1326), range m facing turn card ks[j] in r[m, j]
        -> (M, Ks, 1326).  One matmul per card on the resident win table (a view, no copy), and ONE matmul for all the
        denominators and one `where`, instead of two matmuls and a `where` per (player, card).  Gathering the Ks tables
        into a batch for a single bmm is slower: `wd[ks]` copies 84 MB (18 ms on MPS)."""
        if not self.fast:
            return torch.stack([self.tables[k].equity(r[:, j]) for j, k in enumerate(ks)], dim=1)
        num = torch.stack([r[:, j] @ self.wd[k].T for j, k in enumerate(ks)], dim=1)           # sum_h' wd[k, h, h'] r[h']
        den = (r.reshape(-1, r.shape[-1]) @ self.disjoint.T).reshape(r.shape)
        return torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.full_like(num, 0.5))

    @torch.no_grad()
    def refresh(self, ends, sample=None, net_free=False):
        """ends: [(end node, r0, r1)] -> caches ubar for every leaf; one batched net evaluation.

        sample: indices (into the turn cards) to evaluate this time (chance sampling); None = all.
        net_free: value the leaves with the parameter-free equity baseline instead of the net (no forward pass): what the
        net's output is before any learning, and a cheap way to warm-start the CFR."""
        dev, L, K = self.device, len(ends), len(self.turn_cards)
        ks = list(range(K)) if sample is None else sorted(int(k) for k in sample)
        Ks = len(ks)
        t0 = time.time()
        keep = (~self.blocks).float()[ks]                                                               # (Ks, 1326)
        R0 = torch.tensor(np.stack([e[1] for e in ends]), dtype=torch.float32, device=dev)             # (L, 1326)
        R1 = torch.tensor(np.stack([e[2] for e in ends]), dtype=torch.float32, device=dev)
        rc = torch.cat([R0, R1])[:, None, :] * keep[None]                                              # (2L, Ks, 1326)
        if self.root is not None:
            root2 = torch.cat([self.root[0].expand(L, 1, -1), self.root[1].expand(L, 1, -1)])
            n = mix_with_root(rc, root2, keep[None], self.floor)
        else:
            n = rc / rc.sum(-1, keepdim=True).clamp_min(1e-30)
        n0, n1 = n[:L], n[L:]
        eq = self._equity_all(ks, torch.cat([n1, n0]))                                                  # (2L, Ks, 1326)
        eq_oop, eq_ip = eq[:L], eq[L:]                                                                  # OOP faces n1, IP faces n0
        pot2 = torch.tensor([self.pot + 2 * e[0].contrib[0] for e in ends], dtype=torch.float32, device=dev)
        stack2 = torch.tensor([self.stack - e[0].contrib[0] for e in ends], dtype=torch.float32, device=dev)
        scale = (pot2 + stack2)[:, None, None]                                                          # u = (P' + S') v
        if net_free:
            if dev.type == "mps":
                torch.mps.synchronize()
            t1 = t2 = time.time()
            inv = (1.0 / (1.0 + stack2 / pot2))[:, None, None]                                          # v = (eq - 1/2) / (1 + SPR)
            u_oop, u_ip = scale * inv * (eq_oop - 0.5), scale * inv * (eq_ip - 0.5)                     # (L, Ks, 1326)
        else:
            spr = (stack2 / pot2)[:, None].expand(L, Ks).reshape(-1)
            flat = lambda x: x.reshape(L * Ks, -1)  # noqa: E731
            tile = lambda x: x[ks][None].expand(L, Ks, *x.shape[1:]).reshape(L * Ks, *x.shape[1:])  # noqa: E731
            batch = dict(in_oop=flat(n0), in_ip=flat(n1), f_eq_oop=flat(eq_oop), f_eq_ip=flat(eq_ip), f_eq_unif=tile(self.f_unif),
                         f_cat=tile(self.f_cat), f_outs=tile(self.f_outs), spr=spr, board=tile(self.boards_c),
                         hand_cards=tile(self.hand_cards_c))
            if dev.type == "mps":
                torch.mps.synchronize()
            t1 = time.time()
            v_oop, v_ip = [], []
            m = L * Ks
            step = -(-m // -(-m // self.chunk))            # balanced chunks of at most `chunk` states (60 -> 2 x 30, not 36 + 24)
            for i in range(0, m, step):
                b = {k: v[i:i + step] for k, v in batch.items()}
                if self.support is not None:
                    b["support"] = self.support
                o = self.net(b, policy=False)
                v_oop.append(o["v_oop"])
                v_ip.append(o["v_ip"])
            v_oop, v_ip = torch.cat(v_oop).reshape(L, Ks, -1), torch.cat(v_ip).reshape(L, Ks, -1)
            if dev.type == "mps":
                torch.mps.synchronize()
            t2 = time.time()
            u_oop, u_ip = scale * v_oop, scale * v_ip
            if self.support is not None:                                                                # hands outside: no value
                u_oop, u_ip = (torch.zeros(L, Ks, cards.NUM_COMBOS, device=dev).index_copy_(2, self.support, u) for u in (u_oop, u_ip))
        card_w = None if sample is None else torch.full((Ks,), K / Ks, device=dev)
        self.store(ends, u_oop, u_ip, card_w, R=(R0, R1), blocks=self.blocks[ks])
        self.timing["features"] += t1 - t0
        self.timing["net"] += t2 - t1
        self.timing["post"] += time.time() - t2
        self.timing["refreshes"] += 1


def resolve(net, flop, pot, stack, r0, r1, device, n_iter=100, refresh=10, bet_fracs=(0.5,), raise_fracs=(1.0,),
            max_raises=1, chunk=36, leaves=None, floor=0.003, n_cards=None, seed=0, warm=0):
    """Truncated flop CFR.  Returns dict(game, leaves, timing seconds).  Pass `leaves` to reuse the 49 board tables.

    warm: the first `warm` of the `n_iter` iterations value the leaves with the equity baseline (no net call, ~4x cheaper),
    then the same CFR carries on with the net.  The regrets carry over (they give the starting strategy), the average
    strategy restarts, otherwise it stays biased by the equity leaves.  Off by default: measured on 8 held-out flops it only
    pays at short budgets (20 warm of 60 iterations: 1.08 s, exploitability 0.60 % in the net's game, against ~0.85 % for a
    cold run of the same time; no gain at 100 iterations)."""
    t_all = time.time()
    root = cfr.build_round_tree(pot, stack, bet_fracs, raise_fracs, max_raises)
    leaves = leaves or NetLeaves(net, flop, pot, stack, device, chunk, root=(r0, r1), floor=floor)
    game = cfr.Cfr(root, r0, r1, pot, leaves)
    t_cfr, rng = 0.0, np.random.default_rng(seed)
    for it in range(n_iter):
        if warm and it == warm:
            game.ssum = [np.zeros_like(x) for x in game.ssum]
        if it % refresh == 0:
            leaves.refresh(game.end_reaches(), None if n_cards is None else rng.choice(len(leaves.turn_cards), n_cards, replace=False),
                           net_free=it < warm)
        t0 = time.time()
        game.iterate()
        t_cfr += time.time() - t0
    tm = dict(leaves.timing, tables=leaves.t_tables, cfr=t_cfr, total=time.time() - t_all, n_leaves=len(game.end_reaches()))
    return dict(game=game, leaves=leaves, root=root, timing=tm)
