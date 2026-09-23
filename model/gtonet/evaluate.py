"""Evaluation harness (architecture.md §11): range-weighted MAE in % of pot, sliced for diagnosis.

Predictions and targets are in EV/pot units (the stored `v_*`); weights are the card-removal-corrected marginals
`r_*` (sum 1 per state and player), so a state's error is sum_h c(h) |v_hat(h) - v(h)|, averaged over both players.
Solves, not labels, are the unit: every state counts once.  Never use the zero-sum residual as a metric.
"""
import numpy as np

from . import dataset

SPR_BUCKETS = ((0, 1, "spr<1"), (1, 3, "1<=spr<3"), (3, 8, "3<=spr<8"), (8, 1e9, "spr>=8"))


def state_mae_pct(pred_oop, pred_ip, d) -> np.ndarray:
    """(N,) range-weighted MAE per state, in % of pot."""
    e_oop = (d["r_oop"] * np.abs(pred_oop - d["v_oop"])).sum(1)
    e_ip = (d["r_ip"] * np.abs(pred_ip - d["v_ip"])).sum(1)
    return 100.0 * (e_oop + e_ip) / 2


def slices(d) -> dict:
    tex = np.array([dataset.texture(b) for b in d["board"]])
    spr = d["spr"]
    out = {"all": np.ones(len(spr), bool)}
    for k in sorted(set(d["kind"])):
        out[f"kind={k}"] = d["kind"] == k
    for lo, hi, name in SPR_BUCKETS:
        out[name] = (spr >= lo) & (spr < hi)
    for k in sorted(set(tex)):
        out[f"tex={k}"] = tex == k
    out["perturbed"] = np.asarray(d["perturbed"], bool)
    out["on-policy"] = ~np.asarray(d["perturbed"], bool)
    return out


def report(d, preds: dict, min_n=8) -> str:
    """preds: {name: (pred_oop, pred_ip)} in EV/pot; one table, one column per predictor."""
    sl = slices(d)
    errs = {name: state_mae_pct(po, pi, d) for name, (po, pi) in preds.items()}
    rows = [f"{'slice':<22}{'n':>6}" + "".join(f"{n:>16}" for n in errs)]
    for name, mask in sl.items():
        if mask.sum() < min_n and name != "all":
            continue
        rows.append(f"{name:<22}{int(mask.sum()):>6}" + "".join(f"{e[mask].mean():>15.3f}%" for e in errs.values()))
    return "\n".join(rows)


def equity_baseline(d):
    """The floor of architecture.md §8.6: every hand is worth its (card-removal-corrected) equity, nothing learned."""
    return d["f_eq_oop"], d["f_eq_ip"]


def constant_half(d):
    z = np.full_like(d["v_oop"], 0.5)
    return z, z.copy()


def range_average_oracle(d):
    """Best hand-blind predictor: every hand of a player gets that player's true range-average EV."""
    m_oop = (d["r_oop"] * d["v_oop"]).sum(1, keepdims=True)
    m_ip = (d["r_ip"] * d["v_ip"]).sum(1, keepdims=True)
    return np.repeat(m_oop, d["v_oop"].shape[1], 1), np.repeat(m_ip, d["v_ip"].shape[1], 1)
