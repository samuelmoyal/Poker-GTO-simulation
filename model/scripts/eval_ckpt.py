#!/usr/bin/env python3
"""Evaluate a checkpoint on a split (use `test` only for the final number: it must never guide decisions)."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gtonet import batching, dataset, evaluate, model as M  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import _preds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "net_turn_small.pt"))
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev)
    net = M.NetTurn(M.CONFIGS[ck["config"]]).to(dev)
    net.load_state_dict({k.replace("module.", ""): v for k, v in ck["state"].items() if k != "n_averaged"})
    d = dataset.load(args.split)
    data = batching.TurnData(d, dev)
    print(f"{len(d['id'])} states, {len(set(d['flop_key']))} flops ({args.split}), {os.path.basename(args.ckpt)}")
    print(evaluate.report(d, {"EQUITY floor": evaluate.equity_baseline(d), "net_turn": _preds(net, data)}))


if __name__ == "__main__":
    main()
