#!/usr/bin/env python3
"""Held-out evaluation of the real-time flop resolver (TINY net, 100 iterations, 12 sampled turn cards, one refresh per
iteration) on flops the net never trained on: every val and test flop of the single-raised matchups.

  phase `time` : per spot, wall time of a decision on a NEW flop (49 board tables + 100 iterations) and of a repeat
                 decision on the same flop, and the exploitability of the resolved strategy INSIDE the net's own game
                 (how well the CFR converged on the model; net errors are invisible to it).  Cheap: ~10 s per spot.
  phase `ev`   : REAL EV loss against the exact solver by node locking (bench/real_ev_loss.py), on a few spots.  Each
                 spot needs an exact equilibrium and two locked solves per variant of the labels' reference game:
                 ~7 min each, so this runs for hours.

Variants prune the ranges before resolving: hands whose weight is below `tau` x the range's largest weight are dropped
(weight 0), so the net sees fewer hands.  tau = 0 is the unpruned protocol.

  PYTORCH_ENABLE_MPS_FALLBACK=1 model/.venv/bin/python bench/eval_heldout.py time --taus 0 0.5 0.75
  PYTORCH_ENABLE_MPS_FALLBACK=1 model/.venv/bin/python bench/eval_heldout.py ev --spots 4 --taus 0 0.5
  model/.venv/bin/python bench/eval_heldout.py report
Results: model/data/heldout_time.jsonl, heldout_ev.jsonl (resumable).
"""
import argparse
import glob
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
sys.path.insert(0, os.path.join(HERE, "..", "gto-trainer"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gtonet import cards, cfr, flopdump, labels, model as M, paths, resolver  # noqa: E402
from gto import config, preflop  # noqa: E402
import real_ev_loss as rel  # noqa: E402

DATA = os.path.join(HERE, "..", "model", "data")
TIME_LOG, EV_LOG = os.path.join(DATA, "heldout_time.jsonl"), os.path.join(DATA, "heldout_ev.jsonl")
ITERS, N_CARDS = 100, 12


def spots(split=None):
    """(matchup, flop) of every native flop solve of a single-raised matchup, on the val/test side of the split."""
    out = set()
    for p in glob.glob(os.path.join(paths.NATIVE_FLOPS, "*.json.gz")):
        mk, flop, _ = flopdump.parse_name(p)
        s = cards.split_of(flop)
        if mk in config.MATCHUPS and s != "train" and split in (None, s):
            out.add((mk, "".join(flop), s))
    return sorted(out, key=lambda t: (t[2], t[0], t[1]))


def load_net(name, dev):
    ck = torch.load(os.path.join(HERE, "..", "model", "checkpoints", f"net_turn_{name}.pt"), map_location=dev)
    net = M.NetTurn(M.CONFIGS[ck["config"]]).to(dev).eval()
    net.load_state_dict({k.replace("module.", ""): v for k, v in ck["state"].items() if k != "n_averaged"})
    return net


def setup(mk, flop_s):
    flop = [flop_s[i:i + 2] for i in range(0, 6, 2)]
    m = preflop.get(mk)
    pot, stack = round(m["pot_bb"] * config.SCALE), round(m["stack_bb"] * config.SCALE)
    ip, oop = flopdump.root_reach(mk, flop)
    r0, r1 = (labels._vec(list(d), list(d.values())) for d in (oop, ip))
    return flop, pot, stack, oop, ip, r0, r1


def prune(r, tau):
    return np.where(r >= tau * r.max(), r, 0.0) if tau > 0 else r


def run(net, dev, flop, pot, stack, r0, r1, leaves=None):
    return resolver.resolve(net, flop, float(pot), float(stack), r0, r1, dev, ITERS, 1, bet_fracs=(0.5,), raise_fracs=(),
                            max_raises=0, n_cards=N_CARDS, seed=0, leaves=leaves)


def done(path):
    if not os.path.exists(path):
        return set()
    return {(r["matchup"], r["flop"], r["tau"]) for r in map(json.loads, open(path))}


def phase_time(args):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = load_net(args.net, dev)
    todo = spots()
    seen = done(TIME_LOG)
    print(f"{len(todo)} held-out spots ({sum(1 for s in todo if s[2] == 'val')} val, {sum(1 for s in todo if s[2] == 'test')} test), "
          f"taus {args.taus}, net {args.net} on {dev}", flush=True)
    mk0, fl0, _ = todo[0]
    flop, pot, stack, _, _, r0, r1 = setup(mk0, fl0)
    run(net, dev, flop, pot, stack, r0, r1)              # warm-up: first MPS kernels of the session are slow
    for mk, fl, split in todo:
        flop, pot, stack, oop, ip, r0, r1 = setup(mk, fl)
        base_root = None
        for tau in args.taus:
            if (mk, fl, tau) in seen:
                continue
            p0, p1 = prune(r0, tau), prune(r1, tau)
            res = run(net, dev, flop, pot, stack, p0, p1)
            t = res["timing"]
            game, leaves = res["game"], res["leaves"]
            root = game.avg_sigma(res["root"])                                         # (actions, 1326): OOP root
            w = r0 * cfr.zmass(r1)
            w = w / w.sum()                                                            # weights of the UNPRUNED range
            if base_root is None:
                base_root = root
            res2 = run(net, dev, flop, pot, stack, p0, p1, leaves=leaves)               # same flop again: tables reused
            leaves.chunk = 24
            leaves.refresh(game.end_reaches(strategy=game.avg_sigma))                   # all 49 cards for the read-out
            expl = 100 * game.exploitability() / pot
            row = dict(matchup=mk, flop=fl, split=split, tau=tau, hands=int(len(leaves.support)) if leaves.support is not None else 1326,
                       s_new=t["total"], s_tables=t["tables"], s_features=t["features"], s_net=t["net"], s_post=t["post"], s_cfr=t["cfr"],
                       s_repeat=res2["timing"]["total"], expl_net_game=expl, root_bet=float((w * root[1]).sum()),
                       d_root_vs_base=float((w * np.abs(root[1] - base_root[1])).sum()))
            with open(TIME_LOG, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{split:4s} {mk:10s} {fl} tau {tau:4.2f} hands {row['hands']:4d} | new {row['s_new']:.2f}s repeat {row['s_repeat']:.2f}s "
                  f"| expl(net game) {expl:.2f}% | d_root {row['d_root_vs_base']:.3f}", flush=True)


def phase_ev(args):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = load_net(args.net, dev)
    allspots = spots()
    val = [s for s in allspots if s[2] == "val"]
    test = [s for s in allspots if s[2] == "test"]
    pick = []
    for i in range(args.spots):                          # alternate val / test, spread over matchups
        src = val if i % 2 == 0 else test
        pick.append(src[(i // 2) * max(1, len(src) // max(1, args.spots // 2))])
    seen = done(EV_LOG)
    for mk, fl, split in pick:
        if all((mk, fl, tau) in seen for tau in args.taus):
            continue
        flop, pot, stack, oop, ip, r0, r1 = setup(mk, fl)
        base = {"flop": fl, "pot": pot, "stack": stack, "oop": oop, "ip": ip, "max_iter": 1500, **rel.GAME}
        hands = {0: list(oop), 1: list(ip)}
        print(f"== {split} {mk} {fl}: exact equilibrium ...", flush=True)
        t0 = time.time()
        with labels.LabelSolver(paths.LOCK_BIN, timeout=6 * 3600) as sv:
            ne = sv.solve(dict(base, id="ne", target_pct=args.ne_target))
        print(f"   NE {ne['iters']} it {time.time() - t0:.0f}s expl {ne['expl_pct']:.3f}%", flush=True)
        cands = {}
        for tau in args.taus:
            res = run(net, dev, flop, pot, stack, prune(r0, tau), prune(r1, tau))
            cands[tau] = rel.from_resolver(res["game"], hands)
        local, procs, lock = threading.local(), [], threading.Lock()

        def job(a):
            tau, side, p = a
            if not hasattr(local, "sv"):
                local.sv = labels.LabelSolver(paths.LOCK_BIN, timeout=6 * 3600)
                with lock:
                    procs.append(local.sv)
            locks = [dict(path=path, player=pl, strategy=st) for path, (pl, st) in cands[tau].items() if pl == p]
            r = local.sv.solve(dict(base, id=f"{tau}/{side}", target_pct=args.target, locks=locks))
            print(f"   locked tau {tau} {side}: {r.get('iters')} it {r.get('solve_s', 0):.0f}s expl {r.get('expl_pct', 0):.3f}%", flush=True)
            return tau, side, r
        with ThreadPoolExecutor(args.workers) as ex:
            out = list(ex.map(job, [(tau, side, p) for tau in args.taus for side, p in (("oop", 0), ("ip", 1))]))
        for sv in procs:
            sv.close()
        for tau in args.taus:
            r = {side: rr for t_, side, rr in out if t_ == tau}
            lo, li = ne["ev_oop"] - r["oop"]["ev_oop"], ne["ev_ip"] - r["ip"]["ev_ip"]
            row = dict(matchup=mk, flop=fl, split=split, tau=tau, loss_oop=100 * lo / pot, loss_ip=100 * li / pot,
                       mean=100 * (lo + li) / 2 / pot, ne_expl=ne["expl_pct"], target=args.target)
            with open(EV_LOG, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"   => tau {tau}: real EV loss OOP {row['loss_oop']:.2f}%  IP {row['loss_ip']:.2f}%  mean {row['mean']:.2f}% of pot", flush=True)


def report(args):
    def stats(v):
        v = sorted(v)
        return f"{statistics.mean(v):5.2f} (median {statistics.median(v):5.2f}, p90 {v[int(0.9 * (len(v) - 1))]:5.2f}, max {v[-1]:5.2f})"
    if os.path.exists(TIME_LOG):
        rows = [json.loads(l) for l in open(TIME_LOG)]
        for tau in sorted({r["tau"] for r in rows}):
            for split in ("val", "test", None):
                rs = [r for r in rows if r["tau"] == tau and (split is None or r["split"] == split)]
                if not rs:
                    continue
                print(f"tau {tau:4.2f} {(split or 'all'):4s} n={len(rs):2d} hands {statistics.mean(r['hands'] for r in rs):5.0f} | new flop s {stats([r['s_new'] for r in rs])}"
                      f" | repeat s {stats([r['s_repeat'] for r in rs])}")
                print(f"{'':16s} net-game expl % {stats([r['expl_net_game'] for r in rs])} | net s {statistics.mean(r['s_net'] for r in rs):.2f}"
                      f" feat {statistics.mean(r['s_features'] for r in rs):.2f} post {statistics.mean(r['s_post'] for r in rs):.2f}"
                      f" | root move vs unpruned {statistics.mean(r['d_root_vs_base'] for r in rs):.3f}")
    if os.path.exists(EV_LOG):
        rows = [json.loads(l) for l in open(EV_LOG)]
        print("\nreal EV loss (% of pot; solver noise ~ +-%.2f)" % (rows[0]["target"] / 2))
        for tau in sorted({r["tau"] for r in rows}):
            rs = [r for r in rows if r["tau"] == tau]
            print(f"tau {tau:4.2f} n={len(rs)} mean {statistics.mean(r['mean'] for r in rs):.2f}%  per spot " +
                  "  ".join(f"{r['split'][:1]}:{r['matchup']}/{r['flop']}={r['mean']:.2f}" for r in rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["time", "ev", "report"])
    ap.add_argument("--net", default="tiny_v2")
    ap.add_argument("--taus", type=float, nargs="*", default=[0.0, 0.5, 0.75])
    ap.add_argument("--spots", type=int, default=4, help="ev phase: number of spots (alternating val/test)")
    ap.add_argument("--target", type=float, default=0.3)
    ap.add_argument("--ne-target", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    {"time": phase_time, "ev": phase_ev, "report": report}[args.phase](args)


if __name__ == "__main__":
    main()
