#!/usr/bin/env python3
"""Micro-benchmark of net_turn train steps on MPS (forward+backward+optimizer), and a CPU-fallback check.

  PYTORCH_ENABLE_MPS_FALLBACK=1 model/.venv/bin/python bench/bench_model.py
Any warning mentioning a CPU fallback is a bug (CLAUDE.md): one fallback in the hot loop costs a factor 10.
"""
import os
import sys
import time
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import torch  # noqa: E402
from gtonet import batching, dataset, model as M  # noqa: E402


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = batching.TurnData(dataset.load("train"), dev)
    for name, cfg in (("small", M.SMALL), ("doc", M.DOC)):
        net = M.NetTurn(cfg).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        for bs in (8, 16, 32):
            b = data.batch(list(range(min(bs, data.n))), "canonical")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                times = []
                for i in range(6):
                    t0 = time.time()
                    loss, _ = M.loss_fn(net(b), b)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if dev.type == "mps":
                        torch.mps.synchronize()
                    times.append(time.time() - t0)
            fb = [str(w.message)[:100] for w in caught if "fall" in str(w.message).lower()]
            print(f"{name:6} {M.n_params(net) / 1e6:5.1f}M params  batch {bs:3d}: {1000 * sorted(times[2:])[len(times[2:]) // 2]:7.0f} ms/step"
                  f"  {'FALLBACK: ' + str(fb) if fb else 'no CPU fallback'}")


if __name__ == "__main__":
    main()
