"""Hand engine: deals a heads-up spot, plays the GTO villain, grades the hero's decisions.

Flop strategies come from the precomputed library. Turn and river are re-solved on the fly
with ranges narrowed by the actions that were actually played (class-level weights, because the
solver CLI only accepts hand classes such as `AKs:0.4`, not exact combos).
"""
import json
import os
import random
import threading
import time
import traceback

from . import config, ranges, solver, spots
from .cards import DECK, RVAL, canon, canonical_flop, describe, evaluate, hand_class, map_combo, random_suit_map

STREETS = ("flop", "turn", "river")


# --------------------------------------------------------------------------- tree helpers

def side_of(node) -> str:
    """Player 0 is IP, player 1 is OOP in TexasSolver's dump."""
    return "ip" if node["player"] == 0 else "oop"


def node_strategy(node) -> dict:
    """{canonical combo: [freq per action]} for the acting player (memoised on the node)."""
    s = node.get("_s")
    if s is None:
        s = {canon(k): v for k, v in node["strategy"]["strategy"].items()}
        node["_s"] = s
    return s


def remap_tree(node, suit_map):
    """Rewrite strategy keys of a canonical-suit flop tree into the real suits of this deal."""
    if node.get("node_type") != "action_node":
        return
    node["_s"] = {map_combo(k, suit_map): v for k, v in node["strategy"]["strategy"].items()}
    for child in node.get("childrens", {}).values():
        remap_tree(child, suit_map)


def parse_action(label: str):
    parts = label.split()
    return parts[0].upper(), (float(parts[1]) if len(parts) > 1 else None)


def sample_index(freqs) -> int:
    return random.choices(range(len(freqs)), weights=[max(f, 0.0) for f in freqs], k=1)[0]


def bb(chips) -> float:
    return round(chips / config.SCALE, 2)


# --------------------------------------------------------------------------- persistence / stats

class History:
    def __init__(self, path):
        self.path = path
        self.rows = []
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        self.rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def add(self, row):
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def reset(self):
        self.rows = []
        if os.path.exists(self.path):
            os.rename(self.path, self.path + f".{int(time.time())}.bak")

    def summary(self, matchup=None):
        rows = [r for r in self.rows if not matchup or r["matchup"] == matchup]
        n = len(rows)
        out = dict(decisions=n, score=None, gto_rate=None, by_street={}, by_verdict={}, recent=rows[-15:][::-1])
        if not n:
            return out
        out["score"] = round(100 * sum(r["score"] for r in rows) / n, 1)
        out["gto_rate"] = round(100 * sum(1 for r in rows if r["freq"] >= config.GRADE_OK) / n, 1)
        for st in STREETS:
            sub = [r for r in rows if r["street"] == st]
            if sub:
                out["by_street"][st] = dict(n=len(sub), score=round(100 * sum(r["score"] for r in sub) / len(sub), 1))
        for r in rows:
            out["by_verdict"][r["verdict"]] = out["by_verdict"].get(r["verdict"], 0) + 1
        return out


def grade(freq: float):
    if freq >= config.GRADE_OK:
        verdict = "gto"
    elif freq >= config.GRADE_MIXED:
        verdict = "acceptable"
    elif freq >= config.GRADE_MARGINAL:
        verdict = "marginal"
    else:
        verdict = "erreur"
    return verdict, min(1.0, freq / config.GRADE_OK)


# --------------------------------------------------------------------------- the trainer

class Trainer:
    def __init__(self):
        self.lock = threading.RLock()
        self.h = None            # current hand (dict)
        self.busy = None         # {"msg": str, "progress": {...}} while a solve/job runs
        self.error = None
        self.history = History(os.path.join(config.DATA_DIR, "history.jsonl"))

    # ------------------------------------------------------------------ public API

    def new_hand(self, matchup="random", hero="random", flop_mode="cache"):
        with self.lock:
            if self.busy:
                raise RuntimeError("Un calcul est déjà en cours.")
            if matchup == "random":
                matchup = random.choice(list(config.MATCHUPS))
            if matchup not in config.MATCHUPS:
                raise ValueError(f"matchup inconnu: {matchup}")
            hero_side = hero if hero in ("ip", "oop") else random.choice(["ip", "oop"])
            self.error = None
            self.h = dict(id=int(time.time() * 1000), matchup=matchup, hero_side=hero_side, status="busy",
                          board=[], log=[], feedback=None, result=None, tree=None, node=None)
            self.busy = dict(msg="Préparation de la main…", progress=None)
        self._spawn(self._setup_hand, matchup, hero_side, flop_mode)

    def act(self, index: int):
        with self.lock:
            h = self.h
            if not h or h["status"] != "hero_turn" or self.busy:
                raise RuntimeError("Ce n'est pas ton tour.")
            node = h["node"]
            if not 0 <= index < len(node["actions"]):
                raise ValueError("action invalide")
            self._grade_hero(index)
            h["status"] = "busy"
            if self._apply(index, actor="hero"):
                return
            self.busy = dict(msg="Le bot joue…", progress=None)
        self._spawn(self._continue)

    def reset_stats(self):
        with self.lock:
            self.history.reset()

    # ------------------------------------------------------------------ jobs

    def _spawn(self, fn, *args):
        def run():
            try:
                fn(*args)
            except Exception as exc:  # surface any failure to the UI instead of dying silently
                traceback.print_exc()
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    if self.h:
                        self.h["status"] = "error"
            finally:
                with self.lock:
                    self.busy = None
        threading.Thread(target=run, daemon=True).start()

    def _progress(self, prog):
        with self.lock:
            if self.busy is not None:
                self.busy["progress"] = prog

    def _set_msg(self, msg):
        with self.lock:
            if self.busy is not None:
                self.busy.update(msg=msg, progress=None)

    def _setup_hand(self, matchup, hero_side, flop_mode):
        m = config.MATCHUPS[matchup]
        cached = spots.cached_flops(matchup)
        if flop_mode == "cache" and cached:
            canon_board = list(random.choice(cached)[1])
            self._set_msg("Chargement du flop depuis la bibliothèque…")
        else:
            canon_board = spots.random_canonical_flop()
            self._set_msg("Nouveau flop : résolution en cours (~2 min)…")
        tree = spots.solve_flop(matchup, canon_board, on_progress=self._progress)

        suit_map = random_suit_map()  # canonical suits -> the suits shown to the player
        board = [c[0] + suit_map[c[1]] for c in canon_board]
        board.sort(key=lambda c: RVAL[c[0]], reverse=True)
        remap_tree(tree, suit_map)

        ip_w, oop_w = ranges.load_matchup_ranges(matchup)
        reach = {"ip": ranges.combo_weights(ip_w, board), "oop": ranges.combo_weights(oop_w, board)}
        vill_side = "oop" if hero_side == "ip" else "ip"
        known = {s: self._solved_combos(tree, s) for s in ("ip", "oop")}
        hero_pool = {c: w for c, w in reach[hero_side].items() if c in known[hero_side]}
        hero_cards = ranges.sample_combo(hero_pool)
        dead = set(board) | {hero_cards[:2], hero_cards[2:]}
        vill_pool = {c: w for c, w in reach[vill_side].items()
                     if c in known[vill_side] and c[:2] not in dead and c[2:] not in dead}
        vill_cards = ranges.sample_combo(vill_pool)

        with self.lock:
            h = self.h
            h.update(
                board=board, tree=tree, node=tree, street="flop", reach=reach,
                cards={hero_side: hero_cards, vill_side: vill_cards},
                pot0=m["pot_bb"] * config.SCALE, pot_start=m["pot_bb"] * config.SCALE,
                stack_start=m["stack_bb"] * config.SCALE, contrib={"ip": 0.0, "oop": 0.0},
                invested={"ip": 0.0, "oop": 0.0}, all_in=False, pending_deal=False,
            )
        self._continue()

    @staticmethod
    def _solved_combos(tree, side):
        stack = [tree]
        while stack:
            n = stack.pop()
            if n.get("node_type") != "action_node":
                continue
            if side_of(n) == side:
                return set(node_strategy(n))
            stack.extend(n.get("childrens", {}).values())
        return set()

    def _continue(self):
        """Play villain actions and street transitions until the hero must act or the hand ends."""
        while True:
            with self.lock:
                h = self.h
                if h["status"] == "over":
                    return
                if not h["pending_deal"]:
                    node = h["node"]
                    side = side_of(node)
                    if side == h["hero_side"]:
                        h["status"] = "hero_turn"
                        return
                    strat = node_strategy(node).get(h["cards"][side]) or [1.0] * len(node["actions"])
                    if self._apply(sample_index(strat), actor="villain"):
                        return
                    continue
            self._deal_next_street()

    def _deal_next_street(self):
        with self.lock:
            h = self.h
            street = "turn" if h["street"] == "flop" else "river"
            used = set(h["board"]) | {c for combo in h["cards"].values() for c in (combo[:2], combo[2:])}
            card = random.choice([c for c in DECK if c not in used])
            board = h["board"] + [card]
            classes = {s: ranges.narrow(h["reach"][s], board, keep=[h["cards"][s]]) for s in ("ip", "oop")}
            pot, stack = h["pot_start"], h["stack_start"]
        self._set_msg(f"Résolution du {street} ({''.join(board)})…")
        tree = solver.solve(board, pot, stack, ranges.range_string(classes["ip"]), ranges.range_string(classes["oop"]),
                            street, on_progress=self._progress)
        with self.lock:
            h["board"] = board
            h["street"] = street
            h["tree"] = h["node"] = tree
            h["reach"] = {s: ranges.combo_weights(classes[s], board) for s in ("ip", "oop")}
            h["pending_deal"] = False
            h["contrib"] = {"ip": 0.0, "oop": 0.0}
            self._say("board", street=street, text=f"{street.capitalize()} : {card}")

    # ------------------------------------------------------------------ game logic (call with lock held)

    def _pot_now(self):
        h = self.h
        return h["pot_start"] + h["contrib"]["ip"] + h["contrib"]["oop"]

    def _to_call(self, side):
        h = self.h
        other = "oop" if side == "ip" else "ip"
        return max(0.0, h["contrib"][other] - h["contrib"][side])

    def action_text(self, label, side):
        kind, amt = parse_action(label)
        h = self.h
        remaining = h["stack_start"] - h["contrib"][side]
        if kind == "CHECK":
            return "Check"
        if kind == "FOLD":
            return "Fold"
        if kind == "CALL":
            return f"Call {bb(min(self._to_call(side), remaining))} bb"
        if amt is not None and amt >= h["stack_start"] - 0.5:
            return f"All-in {bb(h['stack_start'])} bb"
        if kind == "BET":
            return f"Miser {bb(amt)} bb ({round(100 * amt / max(self._pot_now(), 1))}% pot)"
        if kind == "RAISE":
            return f"Relancer à {bb(amt)} bb"
        return f"All-in {bb(amt)} bb" if amt else label

    def _say(self, kind, **kw):
        entry = dict(kind=kind, street=self.h.get("street"))
        entry.update(kw)
        self.h["log"].append(entry)

    def _apply(self, idx, actor, freqs=None):
        """Apply action `idx` at the current node. Returns True if the hand is over."""
        h = self.h
        node = h["node"]
        side = side_of(node)
        label = node["actions"][idx]
        kind, amt = parse_action(label)
        text = self.action_text(label, side)
        strat = node_strategy(node)

        # narrow the acting side's reach by the solver's own strategy (not by what was actually chosen)
        reach = h["reach"][side]
        for combo in list(reach):
            f = strat.get(combo)
            reach[combo] = reach[combo] * f[idx] if f else 0.0

        if actor == "villain":
            self._say("villain", side=side, text=text)
        else:
            self._say("hero", side=side, text=text)

        if kind == "FOLD":
            self._finish(fold=side)
            return True
        if kind in ("BET", "RAISE", "ALLIN") and amt is not None:
            h["contrib"][side] = min(amt, h["stack_start"])
        elif kind == "CALL":
            other = "oop" if side == "ip" else "ip"
            h["contrib"][side] = min(h["contrib"][other], h["stack_start"])

        child = node.get("childrens", {}).get(label)
        if child and child.get("node_type") == "action_node":
            h["node"] = child
            return False

        # street is over
        for s in ("ip", "oop"):
            h["invested"][s] += h["contrib"][s]
        h["pot_start"] += h["contrib"]["ip"] + h["contrib"]["oop"]
        h["stack_start"] -= max(h["contrib"].values())
        h["contrib"] = {"ip": 0.0, "oop": 0.0}
        if h["stack_start"] <= 0.5:
            h["all_in"] = True
        if h["street"] == "river" or h["all_in"]:
            self._finish()
            return True
        h["pending_deal"] = True
        return False

    def _finish(self, fold=None):
        h = self.h
        hero, vill = h["hero_side"], ("oop" if h["hero_side"] == "ip" else "ip")
        for s in ("ip", "oop"):
            h["invested"][s] += h["contrib"][s]
        h["pot_final"] = h["pot0"] + h["invested"]["ip"] + h["invested"]["oop"]
        result = dict(pot_bb=bb(h["pot_final"]))
        if fold:
            hero_won = fold != hero
            result.update(kind="fold", text="Le bot se couche" if hero_won else "Tu t'es couché")
        else:
            used = set(h["board"]) | {c for combo in h["cards"].values() for c in (combo[:2], combo[2:])}
            deck = [c for c in DECK if c not in used]
            random.shuffle(deck)
            runout = []
            while len(h["board"]) < 5:
                runout.append(deck.pop())
                h["board"].append(runout[-1])
            if runout:
                self._say("board", text="Runout : " + " ".join(runout))
            sc = {s: evaluate(h["board"] + [h["cards"][s][:2], h["cards"][s][2:]]) for s in ("ip", "oop")}
            if sc[hero] == sc[vill]:
                hero_won = None
            else:
                hero_won = sc[hero] > sc[vill]
            result.update(kind="showdown", hero_hand=describe(sc[hero]), villain_hand=describe(sc[vill]),
                          text="Showdown : égalité" if hero_won is None else ("Showdown : tu gagnes" if hero_won else "Showdown : tu perds"))
        if hero_won is None:
            net = 0.0
        elif hero_won:
            net = h["invested"][vill]
        else:
            net = -h["invested"][hero]
        result.update(hero_won=hero_won, net_bb=bb(net))
        h["result"] = result
        h["status"] = "over"
        self._say("result", text=result["text"])

    # ------------------------------------------------------------------ grading

    def _grade_hero(self, idx):
        h = self.h
        node = h["node"]
        side = h["hero_side"]
        combo = h["cards"][side]
        strat = node_strategy(node)
        freqs = strat.get(combo)
        labels = node["actions"]
        texts = [self.action_text(a, side) for a in labels]
        if freqs is None:
            h["feedback"] = dict(available=False, text="Main absente de la stratégie du solveur (range trop étroite).")
            return
        total = sum(freqs) or 1.0
        freqs = [f / total for f in freqs]
        verdict, score = grade(freqs[idx])
        best = max(range(len(freqs)), key=lambda i: freqs[i])
        reach = h["reach"][side]
        cells = {}
        for c, w in reach.items():
            f = strat.get(c)
            if not f or w <= 0:
                continue
            cls = hand_class(c)
            cell = cells.setdefault(cls, [0.0, [0.0] * len(labels)])
            cell[0] += w
            for i, x in enumerate(f):
                cell[1][i] += w * x
        top = max((c[0] for c in cells.values()), default=1.0) or 1.0
        grid = {cls: dict(w=round(c[0] / top, 3), f=[round(x / c[0], 3) for x in c[1]]) for cls, c in cells.items()}
        overall = [0.0] * len(labels)
        tw = sum(c[0] for c in cells.values()) or 1.0
        for c in cells.values():
            for i, x in enumerate(c[1]):
                overall[i] += x / tw
        h["feedback"] = dict(
            available=True, street=h["street"], hand=combo, hand_class=hand_class(combo),
            actions=[dict(text=t, freq=round(f, 3), kind=parse_action(a)[0]) for t, f, a in zip(texts, freqs, labels)],
            chosen=idx, best=best, freq=round(freqs[idx], 3), verdict=verdict, score=round(score, 3),
            range_freqs=[round(x, 3) for x in overall], grid=grid,
            pot_bb=bb(self._pot_now()), to_call_bb=bb(self._to_call(side)),
        )
        self.history.add(dict(ts=int(time.time()), matchup=h["matchup"], hero_side=side, street=h["street"],
                              board=list(h["board"]), hand=combo, actions=texts, freqs=[round(f, 3) for f in freqs],
                              chosen=idx, freq=round(freqs[idx], 3), verdict=verdict, score=round(score, 3)))

    # ------------------------------------------------------------------ snapshot for the UI

    def snapshot(self):
        with self.lock:
            h = self.h
            snap = dict(busy=self.busy, error=self.error, matchups={k: v["label"] for k, v in config.MATCHUPS.items()},
                        library={k: len(spots.cached_flops(k)) for k in config.MATCHUPS}, hand=None,
                        stats=self.history.summary())
            if h:
                m = config.MATCHUPS[h["matchup"]]
                hero = h["hero_side"]
                vill = "oop" if hero == "ip" else "ip"
                over = h["status"] == "over"
                hand = dict(
                    id=h["id"], status=h["status"], matchup=h["matchup"], label=m["label"],
                    hero_side=hero, hero_pos=m[hero], villain_pos=m[vill], street=h.get("street"),
                    board=h["board"], log=h["log"], result=h["result"], feedback=h["feedback"],
                    hero_cards=[h["cards"][hero][:2], h["cards"][hero][2:]] if h.get("cards") else None,
                    villain_cards=[h["cards"][vill][:2], h["cards"][vill][2:]] if h.get("cards") and over else None,
                    actions=None,
                )
                if h.get("cards"):
                    hand["pot_bb"] = bb(self._pot_now()) if h["result"] is None else bb(h["pot_final"])
                    hand["stack_bb"] = bb(h["stack_start"] - h["contrib"][hero])
                    hand["villain_bet_bb"] = bb(h["contrib"][vill])
                    hand["hero_bet_bb"] = bb(h["contrib"][hero])
                if h["status"] == "hero_turn" and not self.busy:
                    node = h["node"]
                    hand["actions"] = [dict(index=i, text=self.action_text(a, hero), kind=parse_action(a)[0])
                                       for i, a in enumerate(node["actions"])]
                snap["hand"] = hand
            return snap
