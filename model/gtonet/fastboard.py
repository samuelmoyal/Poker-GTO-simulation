"""ctypes front-end of tools/boardlib (Rust): the per-board work the numpy/MPS code did slowly.

  strength(cards)        5-7 card strengths, integer-identical to `poker.strength`
  FastBoards(boards)     n turn boards built in parallel: made-hand category / outs / valid hands, and
                         .equity(ranges) = conditional equity of every hand vs each range (card removal exact),
                         by prefix sums over a sorted order: no 1326x1326 table per board (equity.BoardTables).
"""
import ctypes

import numpy as np
import torch

from . import cards, paths

_lib = None
NH = cards.NUM_COMBOS


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(paths.BOARDLIB)
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        _lib.gt_strength.argtypes = [vp, sz, sz, vp]
        _lib.gt_boards_new.argtypes = [vp, sz, vp]
        _lib.gt_board_free.argtypes = [vp]
        _lib.gt_board_info.argtypes = [vp, vp, vp, vp]
        _lib.gt_equity_multi.argtypes = [vp, sz, vp, sz, vp]
        _lib.gt_boards_wd.argtypes = [vp, sz, vp]
    return _lib


def available() -> bool:
    try:
        lib()
        return True
    except OSError:
        return False


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def strength(hands: np.ndarray) -> np.ndarray:
    """(n, k) card ids, k in 5..7 -> (n,) int32, same encoding as `poker.strength`."""
    a = np.ascontiguousarray(hands, dtype=np.uint8)
    out = np.empty(len(a), dtype=np.int32)
    lib().gt_strength(_ptr(a), len(a), a.shape[1], _ptr(out))
    return out


class FastBoards:
    def __init__(self, boards):
        b = np.ascontiguousarray([[cards.CARD_ID[c] if isinstance(c, str) else int(c) for c in bd] for bd in boards], dtype=np.uint8)
        self.n = len(b)
        self._ptrs = (ctypes.c_void_p * self.n)()
        lib().gt_boards_new(_ptr(b), self.n, ctypes.addressof(self._ptrs))
        self.cat6 = np.empty((self.n, NH), dtype=np.uint8)
        self.outs = np.empty((self.n, NH), dtype=np.uint8)
        self.hand_ok = np.empty((self.n, NH), dtype=np.uint8)
        for j in range(self.n):
            lib().gt_board_info(self._ptrs[j], _ptr(self.cat6[j]), _ptr(self.outs[j]), _ptr(self.hand_ok[j]))
        self.hand_ok = self.hand_ok.astype(bool)

    def equity(self, r: np.ndarray, idx=None) -> np.ndarray:
        """r: (n_boards, m, 1326) raw opponent ranges, one set per board -> (n_boards, m, 1326) float32.

        idx: use only these boards (r then has len(idx) rows), e.g. the sampled turn cards of a refresh."""
        r = np.ascontiguousarray(r, dtype=np.float32)
        ptrs = self._ptrs if idx is None else (ctypes.c_void_p * len(idx))(*[self._ptrs[i] for i in idx])
        n = self.n if idx is None else len(idx)
        assert r.shape[0] == n and r.shape[2] == NH
        out = np.empty_like(r)
        lib().gt_equity_multi(ctypes.addressof(ptrs), n, _ptr(r), r.shape[1], _ptr(out))
        return out

    def uniform_equity(self) -> np.ndarray:
        return self.equity(self.hand_ok.astype(np.float32)[:, None, :])[:, 0, :]

    def win_tables(self) -> np.ndarray:
        """(n, 1326, 1326) float32 WD tables (see equity.BoardTables): equity(r) = (WD r) / (D r).  Built in parallel."""
        out = np.empty((self.n, NH, NH), dtype=np.float32)
        lib().gt_boards_wd(ctypes.addressof(self._ptrs), self.n, _ptr(out))
        return out

    def __del__(self):
        try:
            for p in self._ptrs:
                lib().gt_board_free(p)
        except Exception:  # interpreter shutdown
            pass


class FastBoardTable:
    """Drop-in for `equity.BoardTables` (one board, torch in/out) built on FastBoards."""

    def __init__(self, board, device=None):
        self.fb, self.device = FastBoards([board]), device
        self.cat6, self.outs = self.fb.cat6[0].astype(np.int8), self.fb.outs[0].astype(np.int8)
        self.hand_ok = torch.tensor(self.fb.hand_ok[0], device=device)

    def equity(self, r_opp: torch.Tensor) -> torch.Tensor:
        r = r_opp.detach().cpu().numpy().astype(np.float32)
        flat = r.reshape(1, -1, NH)
        return torch.from_numpy(self.fb.equity(flat).reshape(r.shape)).to(self.device)

    def uniform_equity(self) -> torch.Tensor:
        return torch.from_numpy(self.fb.uniform_equity()[0]).to(self.device)
