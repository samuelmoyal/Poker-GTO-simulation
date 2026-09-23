"""Thin wrapper around TexasSolver's console_solver: build input, run, parse, cache."""
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time

from . import config


class SolverError(RuntimeError):
    pass


_run_lock = threading.Lock()  # one solve at a time: the solver already uses every core


def _fmt(x) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def build_input(board, pot, stack, ip_range, oop_range, street, out_path) -> str:
    """`pot` and `stack` are in solver chips (SCALE per bb).

    `out_path` must contain no spaces (the solver splits commands on them): pass a path relative
    to the solver's working directory.
    """
    effort = config.EFFORT[street]
    lines = [
        f"set_pot {_fmt(pot)}",
        f"set_effective_stack {_fmt(stack)}",
        f"set_board {','.join(board)}",
        f"set_range_ip {ip_range}",
        f"set_range_oop {oop_range}",
    ]
    first = {"flop": 0, "turn": 1, "river": 2}[street]
    for st in ("flop", "turn", "river")[first:]:
        tree = config.BET_TREE[street][st]
        for who in ("oop", "ip"):
            for s in tree["bet"]:
                lines.append(f"set_bet_sizes {who},{st},bet,{s}")
            for s in tree["raise_"]:
                lines.append(f"set_bet_sizes {who},{st},raise,{s}")
            if who == "oop" and st != "flop":
                for s in tree["donk"]:
                    lines.append(f"set_bet_sizes {who},{st},donk,{s}")
            if tree["allin"]:
                lines.append(f"set_bet_sizes {who},{st},allin")
    lines += [
        f"set_allin_threshold {config.ALLIN_THRESHOLD}",
        "build_tree",
        f"set_thread_num {config.THREADS}",
        f"set_accuracy {effort['accuracy']}",
        f"set_max_iteration {effort['iterations']}",
        f"set_print_interval {effort['print_interval']}",
        "set_use_isomorphism 1",
        "start_solve",
        "set_dump_rounds 1",
        f"dump_result {out_path}",
    ]
    return "\n".join(lines) + "\n"


def _read_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _write_gz(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def _progress_from_log(path):
    try:
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", "ignore")
    except OSError:
        return None
    its = re.findall(r"Iter: (\d+)", text)
    expl = re.findall(r"Total exploitability ([0-9.]+)", text)
    return dict(iter=int(its[-1]) if its else 0, exploitability=float(expl[-1]) if expl else None)


def solve(board, pot, stack, ip_range, oop_range, street, cache_path=None, on_progress=None):
    """Solve one street and return the dumped tree (dict). Cached on disk by input hash."""
    probe = build_input(board, pot, stack, ip_range, oop_range, street, "OUT")
    if cache_path is None:
        digest = hashlib.sha1(probe.encode()).hexdigest()
        cache_path = os.path.join(config.STREET_CACHE_DIR, f"{street}_{digest}.json.gz")
    if os.path.exists(cache_path):
        try:
            return _read_gz(cache_path)
        except (OSError, EOFError, json.JSONDecodeError):
            os.remove(cache_path)

    with _run_lock:
        if os.path.exists(cache_path):  # another thread solved it while we waited
            return _read_gz(cache_path)
        os.makedirs(config.WORK_DIR, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="solve_", dir=config.WORK_DIR)
        try:
            out_path = os.path.join(workdir, "out.json")
            log_path = os.path.join(workdir, "solver.log")
            in_path = os.path.join(workdir, "input.txt")
            with open(in_path, "w") as f:
                f.write(build_input(board, pot, stack, ip_range, oop_range, street, "out.json"))
            timeout = config.EFFORT[street]["timeout"]
            with open(log_path, "wb") as log:
                proc = subprocess.Popen(
                    [config.SOLVER_BIN, "--input_file", in_path, "--resource_dir", config.RESOURCE_DIR, "--mode", "holdem"],
                    stdout=log, stderr=subprocess.STDOUT, cwd=workdir, start_new_session=True,
                )
                started = time.time()
                while proc.poll() is None:
                    if on_progress:
                        on_progress(_progress_from_log(log_path))
                    if time.time() - started > timeout:
                        os.killpg(proc.pid, signal.SIGKILL)
                        raise SolverError(f"Solveur trop long (> {timeout}s) pour {street} {board}")
                    time.sleep(0.5)
            if not os.path.exists(out_path):
                tail = open(log_path, "rb").read()[-400:].decode("utf-8", "ignore")
                raise SolverError(f"Le solveur n'a rien produit ({street} {board}): {tail!r}")
            with open(out_path) as f:
                tree = json.load(f)
            _write_gz(cache_path, tree)
            meta = dict(_progress_from_log(log_path) or {}, seconds=round(time.time() - started, 1), street=street, board=list(board))
            with open(cache_path.replace(".json.gz", ".meta.json"), "w") as f:
                json.dump(meta, f)
            return tree
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
