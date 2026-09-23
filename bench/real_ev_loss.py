#!/usr/bin/env python3
"""REAL EV loss of a flop strategy on one spot, measured against the exact solver (node locking).

  loss_p = V_p(equilibrium) - V_p(strategy S of p locked at every flop node, everything else solved exactly)

The reference game is the one `net_turn` was trained on: flop = bet 50% (no raise), turn/river = bets 66% + all-in and
raise 2.5x, all-in thresholds as in the labels.  Candidates: the exact equilibrium itself (control: loss must be ~0),
resolver strategies with different leaf evaluators, and a uniform strategy (scale).

  PYTORCH_ENABLE_MPS_FALLBACK=1 model/.venv/bin/python bench/real_ev_loss.py [--flop 9s7s2s --target 0.3]
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gtonet import cards, flopdump, labels, model as M, paths, resolver  # noqa: E402
from gto import config, preflop  # noqa: E402

GAME = dict(later_bets="66%, a", later_raise="2.5x", add_allin=1.5, force_allin=0.15, donk="")   # the labels' continuation
HERE = os.path.dirname(os.path.abspath(__file__))


def path_of(line):
    return "-".join(t[0] for t in line.split("-") if t)


def from_solver_nodes(nodes, mode="equilibrium"):
    """{path: (player, {hand: {type: prob}})} from the solver's own flop nodes (or uniform / always-check on them)."""
    out = {}
    for path, nd in nodes.items():
        n, acts = len(nd["hands"]), nd["actions"]
        strat = np.array(nd["strategy"]).reshape(len(acts), n)
        d = {}
        for j, h in enumerate(nd["hands"]):
            if mode == "uniform":
                d[h] = {t: 1.0 / len(acts) for t in acts}
            elif mode == "check":
                d[h] = {("x" if "x" in acts else "c"): 1.0}
            else:
                d[h] = {t: float(strat[a, j]) for a, t in enumerate(acts)}
        out[path] = (nd["player"], d)
    return out


def from_resolver(game, hands):
    out = {}
    for node in game.actions:
        s = game.avg_sigma(node)
        types = [a[0] for a in node.actions]
        out[path_of(node.line)] = (node.player, {h: {t: float(s[a, cards.combo_index(h)]) for a, t in enumerate(types)}
                                                 for h in hands[node.player]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flop", default="9s7s2s")
    ap.add_argument("--matchup", default="BTN_vs_BB")
    ap.add_argument("--target", type=float, default=0.3, help="exploitability target (%% pot) of the locked solves")
    ap.add_argument("--ne-target", type=float, default=0.15)
    ap.add_argument("--nets", nargs="*", default=["tiny_v2", "small_v2", "doc_v2"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--cards", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    flop = [args.flop[i:i + 2] for i in range(0, 6, 2)]
    m = preflop.get(args.matchup)
    pot, stack = round(m["pot_bb"] * config.SCALE), round(m["stack_bb"] * config.SCALE)
    ip, oop = flopdump.root_reach(args.matchup, flop)
    base = {"flop": args.flop, "pot": pot, "stack": stack, "oop": oop, "ip": ip, "max_iter": 1500, **GAME}
    hands = {0: list(oop), 1: list(ip)}
    r0, r1 = (labels._vec(list(d), list(d.values())) for d in (oop, ip))
    print(f"{args.matchup} {args.flop}: pot {pot / config.SCALE:g}bb, {len(oop)}/{len(ip)} hands | reference: flop 50% no raise, "
          f"turn/river 66%+all-in+raise 2.5x", flush=True)

    # 1) exact equilibrium (in a thread) while the resolvers run on the GPU
    ne = {}

    def solve_ne():
        with labels.LabelSolver(paths.LOCK_BIN, timeout=3600) as sv:
            ne.update(sv.solve(dict(base, id="ne", target_pct=args.ne_target)))
    th = threading.Thread(target=solve_ne)
    th.start()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cands = {}
    for name in args.nets + ["equity-only"]:
        if name == "equity-only":
            net = M.NetTurn(M.TINY).to(dev).eval()       # zero-initialised heads: leaf value = the parameter-free equity baseline
        else:
            ck = torch.load(os.path.join(HERE, "..", "model", "checkpoints", f"net_turn_{name}.pt"), map_location=dev)
            net = M.NetTurn(M.CONFIGS[ck["config"]]).to(dev).eval()
            net.load_state_dict({k.replace("module.", ""): v for k, v in ck["state"].items() if k != "n_averaged"})
        t0 = time.time()
        res = resolver.resolve(net, flop, float(pot), float(stack), r0, r1, dev, args.iters, 1, bet_fracs=(0.5,), raise_fracs=(),
                               max_raises=0, n_cards=args.cards)
        cands[f"resolver [{name}]"] = from_resolver(res["game"], hands)
        print(f"  resolver with {name}: {time.time() - t0:.1f}s", flush=True)
    th.join()
    if not ne.get("ok"):
        sys.exit(f"equilibrium solve failed: {ne}")
    print(f"exact equilibrium: {ne['iters']} it, {ne['solve_s']:.0f}s, exploitability {ne['expl_pct']:.3f}% | EV oop {ne['ev_oop']:.3f} "
          f"ip {ne['ip'] if False else ne['ev_ip']:.3f} (sum {ne['ev_oop'] + ne['ev_ip']:.3f} = pot)", flush=True)
    cands = {"exact equilibrium (control)": from_solver_nodes(ne["nodes"]), **cands,
             "uniform (scale)": from_solver_nodes(ne["nodes"], "uniform")}

    # 2) two locked solves per candidate (lock OOP's flop strategy / lock IP's), run on `workers` solver processes
    local = threading.local()
    procs, lock = [], threading.Lock()

    def run(job):
        name, side, locks = job
        if not hasattr(local, "sv"):
            local.sv = labels.LabelSolver(paths.LOCK_BIN, timeout=3600)
            with lock:
                procs.append(local.sv)
        r = local.sv.solve(dict(base, id=f"{name}/{side}", target_pct=args.target, locks=locks))
        print(f"  solved {name} / {side} locked: {r.get('iters')} it, {r.get('solve_s', 0):.0f}s, expl {r.get('expl_pct', 0):.3f}%"
              f"{'' if r['ok'] else ' ERROR ' + str(r.get('error'))}", flush=True)
        return name, side, r

    jobs = []
    for name, cand in cands.items():
        for side, p in (("oop", 0), ("ip", 1)):
            jobs.append((name, side, [dict(path=path, player=pl, strategy=st) for path, (pl, st) in cand.items() if pl == p]))
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(run, jobs))
    for sv in procs:
        sv.close()

    print(f"\nREAL EV loss vs the exact solver ({args.matchup} {args.flop}, pot {pot / config.SCALE:g} bb; solver noise ~ +-{args.target / 2:.2f}% pot)")
    print(f"{'flop strategy':34s}{'loss OOP':>10s}{'loss IP':>10s}{'mean % pot':>12s}{'bb per hand':>13s}{'bb/100':>9s}")
    rows = {}
    for name, side, r in results:
        rows.setdefault(name, {})[side] = r
    dump = {}
    for name, d in rows.items():
        lo = ne["ev_oop"] - d["oop"]["ev_oop"]
        li = ne["ev_ip"] - d["ip"]["ev_ip"]
        mean = (lo + li) / 2
        dump[name] = dict(loss_oop_pct=100 * lo / pot, loss_ip_pct=100 * li / pot, mean_pct=100 * mean / pot, bb=mean / config.SCALE)
        print(f"{name:34s}{100 * lo / pot:9.2f}%{100 * li / pot:9.2f}%{100 * mean / pot:11.2f}%{mean / config.SCALE:13.3f}{100 * mean / config.SCALE:9.1f}")
    with open(os.path.join(HERE, "..", "model", "data", "real_ev_loss.json"), "w") as f:
        json.dump(dict(spot=f"{args.matchup} {args.flop}", ne=dict(ev_oop=ne["ev_oop"], ev_ip=ne["ev_ip"], expl_pct=ne["expl_pct"]), losses=dump), f, indent=1)


if __name__ == "__main__":
    main()
