//! Solve one street (flop, turn or river root) and dump the solved betting tree of THAT street with every hand's
//! strategy, in the layout `gto-trainer/gto/engine.py` reads (the layout of TexasSolver's `dump_result`, so the trainer
//! does not care which solver produced it).
//!
//! Differences from `flop_lock` / `turn-labels`: the tree is exported in full (not by action-type path), the actions
//! carry chip amounts ("BET 86.000000", "RAISE 541.000000": totals put in on this street, as TexasSolver prints
//! them), and player 0 is IP (TexasSolver's convention; postflop-solver's is 0 = OOP).
//!
//! request (one JSON per line; sizes are postflop-solver strings, per-street, same for both players):
//!   {"id","board":"AhKd7c[2s[9h]]","pot","stack","oop":{hand:w},"ip":{hand:w},
//!    "flop":{"bet":"50%","raise":"100%","donk":""},"turn":{..},"river":{..},
//!    "target_pct":0.5,"max_iter":1000,"max_seconds":60,"max_gb":8,"add_allin":1.5,"force_allin":0.15,
//!    "progress":false}
//! stdout: optional {"progress":{"iter","expl_pct","seconds"}} lines, then
//!   {"id","ok":true,"iters","expl_pct","solve_s","mem_gb","ev_oop","ev_ip","tree":{..}}  or {"id","ok":false,"error"}
use postflop_solver::*;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

fn d_target() -> f32 { 0.5 }
fn d_iters() -> u32 { 1000 }
fn d_seconds() -> f64 { 120.0 }
fn d_gb() -> f64 { 8.0 }
fn d_add_allin() -> f64 { 1.5 }
fn d_force_allin() -> f64 { 0.15 }
fn d_bet() -> String { "66%, a".into() }
fn d_raise() -> String { "2.5x".into() }

#[derive(Deserialize, Clone)]
struct Sizes {
    #[serde(default = "d_bet")] bet: String,
    #[serde(default = "d_raise")] raise: String,
    #[serde(default)] donk: String,
}

impl Default for Sizes {
    fn default() -> Self { Sizes { bet: d_bet(), raise: d_raise(), donk: String::new() } }
}

#[derive(Deserialize)]
struct Request {
    id: String,
    board: String,
    pot: i32,
    stack: i32,
    oop: HashMap<String, f32>,
    ip: HashMap<String, f32>,
    #[serde(default)] flop: Sizes,
    #[serde(default)] turn: Sizes,
    #[serde(default)] river: Sizes,
    #[serde(default = "d_target")] target_pct: f32,
    #[serde(default = "d_iters")] max_iter: u32,
    #[serde(default = "d_seconds")] max_seconds: f64,
    #[serde(default = "d_gb")] max_gb: f64,
    #[serde(default = "d_add_allin")] add_allin: f64,
    #[serde(default = "d_force_allin")] force_allin: f64,
    #[serde(default)] progress: bool,
}

fn build_range(m: &HashMap<String, f32>) -> Result<Range, String> {
    let (mut hands, mut weights) = (Vec::new(), Vec::new());
    for (h, &w) in m {
        if w <= 0.0 || h.len() != 4 { continue; }
        hands.push((card_from_str(&h[0..2])?, card_from_str(&h[2..4])?));
        weights.push(w.min(1.0));
    }
    Range::from_hands_weights(&hands, &weights)
}

fn options(s: &Sizes) -> Result<(BetSizeOptions, Option<DonkSizeOptions>), String> {
    let bets = BetSizeOptions::try_from((s.bet.as_str(), s.raise.as_str()))?;
    let donk = if s.donk.is_empty() { None } else { Some(DonkSizeOptions::try_from(s.donk.as_str())?) };
    Ok((bets, donk))
}

fn r5(x: f32) -> f64 { ((x as f64) * 1e5).round() / 1e5 }

/// Order actions like TexasSolver does (check/call first, then bets by size, fold last) and label them.
fn label_actions(acts: &[Action]) -> (Vec<usize>, Vec<String>) {
    let facing_bet = acts.iter().any(|a| matches!(a, Action::Fold));
    let key = |a: &Action| -> (u8, i32) {
        match *a {
            Action::Check | Action::Call => (0, 0),
            Action::Bet(x) | Action::Raise(x) | Action::AllIn(x) => (1, x),
            Action::Fold => (2, 0),
            _ => (3, 0),
        }
    };
    let mut order: Vec<usize> = (0..acts.len()).collect();
    order.sort_by_key(|&i| key(&acts[i]));
    let labels = order.iter().map(|&i| match acts[i] {
        Action::Check => "CHECK".to_string(),
        Action::Call => "CALL".to_string(),
        Action::Fold => "FOLD".to_string(),
        Action::Bet(x) => format!("BET {x}.000000"),
        Action::Raise(x) => format!("RAISE {x}.000000"),
        Action::AllIn(x) => format!("{} {x}.000000", if facing_bet { "RAISE" } else { "BET" }),
        _ => "?".to_string(),
    }).collect();
    (order, labels)
}

fn build(game: &mut PostFlopGame, hist: &mut Vec<usize>) -> Value {
    game.apply_history(hist);
    if game.is_terminal_node() { return json!({"node_type": "terminal_node"}); }
    if game.is_chance_node() { return json!({"node_type": "chance_node", "deal_number": 0}); }
    let acts = game.available_actions();
    let p = game.current_player();
    let hands = holes_to_strings(game.private_cards(p)).unwrap();
    let n = hands.len();
    let strat = game.strategy();
    let (order, labels) = label_actions(&acts);

    let mut per_hand = Map::new();
    for (j, h) in hands.iter().enumerate() {
        let v: Vec<f64> = order.iter().map(|&a| r5(strat[a * n + j])).collect();
        per_hand.insert(h.clone(), json!(v));
    }
    let mut children = Map::new();
    for (k, &a) in order.iter().enumerate() {
        if matches!(acts[a], Action::Fold) { continue; }
        hist.push(a);
        let child = build(game, hist);
        hist.pop();
        children.insert(labels[k].clone(), child);
    }
    json!({
        "node_type": "action_node",
        "player": 1 - p,
        "actions": labels,
        "strategy": {"actions": labels, "strategy": per_hand},
        "childrens": children,
    })
}

fn solve_one(req: &Request, out: &mut impl Write) -> Result<Value, String> {
    let b = req.board.as_str();
    if !(b.len() == 6 || b.len() == 8 || b.len() == 10) { return Err(format!("bad board {b:?}")); }
    let card_config = CardConfig {
        range: [build_range(&req.oop)?, build_range(&req.ip)?],
        flop: flop_from_str(&b[0..6])?,
        turn: if b.len() >= 8 { card_from_str(&b[6..8])? } else { NOT_DEALT },
        river: if b.len() == 10 { card_from_str(&b[8..10])? } else { NOT_DEALT },
    };
    let (fb, _) = options(&req.flop)?;
    let (tb, td) = options(&req.turn)?;
    let (rb, rd) = options(&req.river)?;
    let tree_config = TreeConfig {
        initial_state: match b.len() { 6 => BoardState::Flop, 8 => BoardState::Turn, _ => BoardState::River },
        starting_pot: req.pot, effective_stack: req.stack, rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [fb.clone(), fb], turn_bet_sizes: [tb.clone(), tb], river_bet_sizes: [rb.clone(), rb],
        turn_donk_sizes: td, river_donk_sizes: rd,
        add_allin_threshold: req.add_allin, force_allin_threshold: req.force_allin, merging_threshold: 0.1,
    };
    let mut game = PostFlopGame::with_config(card_config, ActionTree::new(tree_config)?)?;
    let mem_gb = game.memory_usage().0 as f64 / 1e9;
    if mem_gb > req.max_gb { return Err(format!("tree needs {mem_gb:.1} GB > limit {}", req.max_gb)); }
    game.allocate_memory(false);

    let t0 = Instant::now();
    let target = req.pot as f32 * req.target_pct / 100.0;
    let mut expl = compute_exploitability(&game);
    let mut iters = 0u32;
    while iters < req.max_iter && expl > target && t0.elapsed().as_secs_f64() < req.max_seconds {
        solve_step(&game, iters);
        iters += 1;
        if iters % 10 == 0 || iters == req.max_iter {
            expl = compute_exploitability(&game);
            if req.progress {
                let line = json!({"progress": {"iter": iters, "expl_pct": 100.0 * expl as f64 / req.pot as f64,
                                               "seconds": t0.elapsed().as_secs_f64()}});
                writeln!(out, "{line}").unwrap();
                out.flush().unwrap();
            }
        }
    }
    finalize(&mut game);
    let solve_s = t0.elapsed().as_secs_f64();

    game.back_to_root();
    game.cache_normalized_weights();
    let (ev0, ev1) = (game.expected_values(0), game.expected_values(1));
    let (w0, w1) = (game.normalized_weights(0).to_vec(), game.normalized_weights(1).to_vec());
    let (ev_oop, ev_ip) = (compute_average(&ev0, &w0) as f64, compute_average(&ev1, &w1) as f64);
    let tree = build(&mut game, &mut Vec::new());
    Ok(json!({"iters": iters, "expl_pct": 100.0 * expl as f64 / req.pot as f64, "solve_s": solve_s, "mem_gb": mem_gb,
              "ev_oop": ev_oop, "ev_ip": ev_ip, "tree": tree}))
}

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }
        let resp = match serde_json::from_str::<Request>(&line) {
            Err(e) => json!({"id": "", "ok": false, "error": format!("bad request: {e}")}),
            Ok(req) => {
                let r = catch_unwind(AssertUnwindSafe(|| solve_one(&req, &mut out)));
                match r {
                    Ok(Ok(mut d)) => {
                        let m = d.as_object_mut().unwrap();
                        m.insert("id".into(), json!(req.id));
                        m.insert("ok".into(), json!(true));
                        d
                    }
                    Ok(Err(e)) => json!({"id": req.id, "ok": false, "error": e}),
                    Err(_) => json!({"id": req.id, "ok": false, "error": "solver panicked"}),
                }
            }
        };
        writeln!(out, "{resp}").unwrap();
        out.flush().unwrap();
    }
}
