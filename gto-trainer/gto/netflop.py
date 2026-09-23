"""Flop solved by the value network ("main IRL"): a truncated CFR over the flop betting tree whose leaves are turn
roots valued by `net_turn` (model/gtonet, architecture.md §12), returned in the tree layout `engine.py` reads.

No pre-solved flop is needed: the input is the board and the two ranges.  The turn and the river are then solved
exactly by the trainer as usual.  Needs torch (run the server with `model/.venv/bin/python`).

The flop game is the one the network was trained for: one bet size (`FLOP_BET`), no raise, no all-in.  Environment:
  GTO_NET_CKPT    checkpoint name in model/checkpoints (default tiny_v2; small_v2 is more accurate and slower)
  GTO_NET_DEVICE  mps (default when available: 2.7-3.3 s per flop on an M2) or cpu (3.5-5.4 s)
  GTO_NET_ITERS   CFR iterations (default 100)
"""
import os
import sys
import threading

from . import config

MODEL_DIR = os.path.join(os.path.dirname(config.ROOT), "model")
FLOP_BET = 0.5
N_CARDS = 12
_lock = threading.Lock()
_state = {}


class Unavailable(RuntimeError):
    pass


def _load():
    if _state:
        return _state
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    try:
        import torch
        from gtonet import cards, cfr, model as M, resolver
    except ImportError as exc:
        raise Unavailable(f"réseau indisponible ({exc}) : lance le serveur avec model/.venv/bin/python") from exc
    name = os.environ.get("GTO_NET_CKPT", "tiny_v2")
    dev = torch.device(os.environ.get("GTO_NET_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("GTO_NET_THREADS", "4")))
    path = os.path.join(MODEL_DIR, "checkpoints", f"net_turn_{name}.pt")
    if not os.path.exists(path):
        raise Unavailable(f"checkpoint introuvable: {path}")
    ck = torch.load(path, map_location=dev)
    net = M.NetTurn(M.CONFIGS[ck["config"]]).to(dev).eval()
    net.load_state_dict({k.replace("module.", ""): v for k, v in ck["state"].items() if k != "n_averaged"})
    _state.update(torch=torch, cards=cards, cfr=cfr, resolver=resolver, net=net, dev=dev, name=name,
                  iters=int(os.environ.get("GTO_NET_ITERS", "100")))
    return _state


def preload():
    """Load the checkpoint (and warm the device) so the first hand does not pay for it."""
    with _lock:
        _load()


def _vector(st, reach):
    v = st["torch"].zeros(st["cards"].NUM_COMBOS)
    for hand, w in reach.items():
        v[st["cards"].combo_index(hand)] = w
    return v.numpy().astype("float64")


def _label(kind, amount=None):
    return kind if amount is None else f"{kind} {int(round(amount))}.000000"


def to_tree(game, node, hands, st):
    """cfr.Node -> engine tree: player 0 = IP there (0 = OOP in `cfr`), actions ordered check/call, bets, fold."""
    if node.kind != "action":
        return {"node_type": "chance_node", "deal_number": 0}
    player = node.player
    facing = node.actions[0] == "f"
    order = [i for i, a in enumerate(node.actions) if a not in ("f",)]
    if facing:
        order = [order[0]] + sorted(order[1:], key=lambda i: node.children[i].contrib[player]) + [0]   # call, raises, fold
    labels = []
    for i in order:
        a = node.actions[i]
        labels.append({"x": "CHECK", "c": "CALL", "f": "FOLD"}.get(a) or _label("RAISE" if facing else "BET", node.children[i].contrib[player]))
    sigma = game.avg_sigma(node)                                                        # (actions, 1326)
    strategy = {h: [round(float(sigma[i, st["cards"].combo_index(h)]), 5) for i in order] for h in hands[player]}
    children = {lab: to_tree(game, node.children[i], hands, st) for lab, i in zip(labels, order) if node.actions[i] != "f"}
    return {"node_type": "action_node", "player": 1 - player, "actions": labels,
            "strategy": {"actions": labels, "strategy": strategy}, "childrens": children}


def solve(board, pot, stack, ip_reach, oop_reach):
    """Solve the flop `board` (3 cards) for the ranges {combo: weight}; pot and stack in solver chips."""
    with _lock:
        st = _load()
        cards, resolver, cfr = st["cards"], st["resolver"], st["cfr"]
        r0, r1 = _vector(st, oop_reach), _vector(st, ip_reach)
        res = resolver.resolve(st["net"], list(board), float(pot), float(stack), r0, r1, st["dev"], st["iters"], 1,
                               bet_fracs=(FLOP_BET,), raise_fracs=(), max_raises=0, n_cards=N_CARDS, seed=0)
        hands = {0: [h for h, w in oop_reach.items() if w > 0], 1: [h for h, w in ip_reach.items() if w > 0]}
        return to_tree(res["game"], res["root"], hands, st)
