"""Range-sensitivity test (architecture.md §11): swap the opponent's range for a very different one and check the
prediction moves like the solver's does.  A net that learned "average value given the board" and ignores ranges
scores fine in training and is useless at runtime, where the range is the variable that changes between solves.
"""
import numpy as np
import torch

from . import cards, equity, model as M


def swapped_ranges(f_eq_unif: np.ndarray, in_opp: np.ndarray):
    """Linear (top half) and polarised (top 15% + bottom 15%) versions of the opponent's range, by hand strength."""
    live = np.flatnonzero(in_opp > 0)
    order = live[np.argsort(-f_eq_unif[live])]
    n = len(order)
    lin, pol = np.zeros_like(in_opp), np.zeros_like(in_opp)
    lin[order[: n // 2]] = 1.0
    pol[order[: max(1, int(0.15 * n))]] = 1.0
    pol[order[n - max(1, int(0.15 * n)):]] = 1.0
    return lin / lin.sum(), pol / pol.sum()


@torch.no_grad()
def predict_variant(net, board, spr, in_oop, in_ip, device) -> dict:
    """The net's EV/pot for both sides on a (possibly modified) state, in the original hand order.

    Features are recomputed exactly for the new ranges (they are range-dependent), exactly as at runtime.
    """
    from . import batching
    board = [int(c) for c in board]
    f = equity.features(board, in_oop, in_ip, device)
    t = lambda x, dt=torch.float32: torch.tensor(np.asarray(x)[None], dtype=dt, device=device)  # noqa: E731
    b = dict(in_oop=t(in_oop), in_ip=t(in_ip), spr=torch.tensor([float(spr)], device=device),
             f_eq_unif=t(f["f_eq_unif"]), f_eq_oop=t(f["f_eq_oop"]), f_eq_ip=t(f["f_eq_ip"]),
             f_cat=t(f["f_cat"], torch.long), f_outs=t(f["f_outs"]))
    canon = torch.tensor(batching.canonical_perm_ids([board]), device=device)
    b = {k: batching.apply_perm(v, canon) if k not in ("spr",) else v for k, v in b.items()}
    b["board"] = torch.gather(batching.CARD_PERMS.to(device)[canon], 1, torch.tensor([board], device=device))
    net.eval()
    o = net(b)
    return {s: batching.unapply_perm(M.to_evp(o[f"v_{s}"], b["spr"]), canon)[0].float().cpu().numpy() for s in ("oop", "ip")} | \
           {"eq_oop": f["f_eq_oop"], "eq_ip": f["f_eq_ip"]}


def range_sensitivity(net, d: dict, i: int, device, side="oop") -> dict:
    """Model-only version: mean |v_hat(linear opp) - v_hat(polarised opp)| over the acting side's range, % of pot."""
    in_oop, in_ip = np.array(d["in_oop"][i]), np.array(d["in_ip"][i])
    opp = in_ip if side == "oop" else in_oop
    lin, pol = swapped_ranges(np.array(d["f_eq_unif"][i]), opp)
    outs = []
    for r in (lin, pol):
        a, b = (in_oop, r) if side == "oop" else (r, in_ip)
        outs.append(predict_variant(net, d["board"][i], d["spr"][i], a, b, device)[side])
    c = in_oop if side == "oop" else in_ip
    return dict(mean_abs_shift_pct=100.0 * float((c * np.abs(outs[0] - outs[1])).sum() / c.sum()))
