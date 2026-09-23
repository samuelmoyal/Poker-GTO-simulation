#!/usr/bin/env python3
"""Time one real-time flop resolve (truncated DCFR + net_turn leaves) and break the time down.

  PYTORCH_ENABLE_MPS_FALLBACK=1 model/.venv/bin/python bench/bench_resolve.py --ckpt model/checkpoints/net_turn_small_v2.pt

Machine must be idle for the numbers to mean anything.  The root-strategy comparison against TexasSolver is a
sanity check only (different flop trees: no re-raise, no all-in here), not a validation.
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gtonet import cards, flopdump, model as M, paths, resolver  # noqa: E402
from gto import config, preflop  # noqa: E402


def vec(d):
    v = np.zeros(cards.NUM_COMBOS)
    for h, w in d.items():
        v[cards.combo_index(h)] = w
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "checkpoints", "net_turn_small.pt"))
    ap.add_argument("--matchup", default="BTN_vs_BB")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--refresh", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=36)
    ap.add_argument("--cards", type=int, default=None, help="chance sampling: turn cards evaluated per refresh (default all 49)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=dev)
    net = M.NetTurn(M.CONFIGS[ck["config"]]).to(dev).eval()
    net.load_state_dict({k.replace("module.", ""): v for k, v in ck["state"].items() if k != "n_averaged"})

    # a validation flop that has a TexasSolver solution (for the sanity comparison)
    tex = [p for p in sorted(glob.glob(os.path.join(paths.FLOP_CACHE, f"{args.matchup}__*.json.gz")))
           if cards.split_of(flopdump.parse_name(p)[1]) == "val"]
    path = tex[0]
    _, flop, _ = flopdump.parse_name(path)
    m = preflop.get(args.matchup)
    pot, stack = m["pot_bb"] * config.SCALE, m["stack_bb"] * config.SCALE
    ip, oop = flopdump.root_reach(args.matchup, flop)
    r0, r1 = vec(oop), vec(ip)
    print(f"{args.matchup} flop {''.join(flop)} (val) | pot {pot:g} stack {stack:g} | net {os.path.basename(args.ckpt)} "
          f"({M.n_params(net) / 1e6:.2f}M) on {dev} | {args.iters} iterations, refresh every {args.refresh}")

    res, prev = None, {}
    for rep in range(args.repeat):
        # 2nd run reuses the 49 board tables: what a second decision on the same flop costs
        res = resolver.resolve(net, flop, pot, stack, r0, r1, dev, args.iters, args.refresh, chunk=args.chunk, n_cards=args.cards,
                               leaves=None if rep == 0 else res["leaves"])
        t = res["timing"]
        d = {k: t[k] - prev.get(k, 0.0) for k in ("features", "net", "post", "refreshes")}   # this run only (leaves accumulate)
        prev = {k: t[k] for k in ("features", "net", "post", "refreshes")}
        tables = t["tables"] if rep == 0 else 0.0
        print(f"run {rep + 1}: total {t['total']:.2f}s = board tables {tables:.2f} + features {d['features']:.2f} + "
              f"net {d['net']:.2f} + post {d['post']:.2f} + CFR loop {t['cfr']:.2f}  "
              f"({int(d['refreshes'])} refreshes x {t['n_leaves']} leaves x {args.cards or 49} cards)", flush=True)

    # exploitability INSIDE the net's game: leaf values frozen at the average strategy's ranges, all turn cards evaluated.
    # This is how well the CFR converged on the model, NOT the strategy's real exploitability (net errors are invisible here).
    res["leaves"].refresh(res["game"].end_reaches(strategy=res["game"].avg_sigma))
    print(f"exploitability in the net's own game: {100 * res['game'].exploitability() / pot:.2f}% of pot")
    sup = res["leaves"].support
    print(f"hands evaluated per state: {'all 1326' if sup is None else f'{len(sup)} of 1326 (initial range)'}")
    game, root = res["game"], res["root"]
    s = game.avg_sigma(root)                                    # OOP root: [check, bet]
    w = (r0 * cfr_zmass(r1)); w = w / w.sum()
    print("root OOP mix (range-weighted):  " + "  ".join(f"{a}={float((w * s[i]).sum()):.3f}" for i, a in enumerate(root.actions)))
    import gto.solver as gs  # noqa: F401
    tree = flopdump.load(path)
    acts = tree["strategy"]["actions"]
    bet_i = next(i for i, a in enumerate(acts) if a.startswith("BET"))
    ts = tree["strategy"]["strategy"]
    idx = [cards.combo_index(h) for h in ts]
    tex_bet = np.zeros(cards.NUM_COMBOS)
    tex_bet[idx] = [p[bet_i] for p in ts.values()]
    print(f"TexasSolver root:  bet={float((w * tex_bet).sum()):.3f} | mean |bet_mine - bet_texas| per hand "
          f"{float((w * np.abs(s[1] - tex_bet)).sum()):.3f}")


def cfr_zmass(r):
    from gtonet import cfr
    return cfr.zmass(r)


if __name__ == "__main__":
    main()
