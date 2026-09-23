//! JSONL in (stdin) -> JSONL out (stdout): one turn-root state per line, solved exactly through the river.
//!
//! request : {"id","flop":"Td9d6h","turn":"Qc","pot":120,"stack":940,
//!            "oop":{"AhKs":0.4,..},"ip":{..}, "target_pct":0.5,"max_iter":2000,"bets":"66%, a","raise":"2.5x"}
//! response: {"id","ok":true,"iters","expl_pct","solve_s","ev_sum_err","oop_hands","ip_hands","oop_w","ip_w",
//!            "oop_ev","ip_ev","root_actions","root_strategy"}   (EV in chips; strategy is [action][hand] flattened)
use postflop_solver::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

fn d_target() -> f32 { 0.5 }
fn d_iters() -> u32 { 2000 }
fn d_bets() -> String { "66%, a".into() }
fn d_raise() -> String { "2.5x".into() }
fn d_add_allin() -> f64 { 1.5 }
fn d_force_allin() -> f64 { 0.15 }

#[derive(Deserialize)]
struct Request {
    id: String,
    flop: String,
    turn: String,
    pot: i32,
    stack: i32,
    oop: HashMap<String, f32>,
    ip: HashMap<String, f32>,
    #[serde(default = "d_target")] target_pct: f32,
    #[serde(default = "d_iters")] max_iter: u32,
    #[serde(default = "d_bets")] bets: String,
    #[serde(default = "d_raise")] raise: String,
    /// optional: start at the river instead of the turn (flop + turn + river all given)
    #[serde(default)] river: Option<String>,
    #[serde(default = "d_add_allin")] add_allin: f64,
    #[serde(default = "d_force_allin")] force_allin: f64,
}

#[derive(Serialize, Default)]
struct Solved {
    iters: u32,
    expl_pct: f64,
    solve_s: f64,
    ev_sum_err: f64,
    oop_hands: Vec<String>,
    ip_hands: Vec<String>,
    oop_w: Vec<f64>,
    ip_w: Vec<f64>,
    oop_ev: Vec<f64>,
    ip_ev: Vec<f64>,
    root_actions: Vec<String>,
    root_strategy: Vec<f64>,
}

#[derive(Serialize)]
struct Response {
    id: String,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(flatten)]
    data: Option<Solved>,
}

fn r5(x: f32) -> f64 { ((x as f64) * 1e5).round() / 1e5 }
fn r5s(v: &[f32]) -> Vec<f64> { v.iter().map(|&x| r5(x)).collect() }

fn build_range(m: &HashMap<String, f32>) -> Result<Range, String> {
    let (mut hands, mut weights) = (Vec::new(), Vec::new());
    for (h, &w) in m {
        if w <= 0.0 || h.len() != 4 { continue; }
        hands.push((card_from_str(&h[0..2])?, card_from_str(&h[2..4])?));
        weights.push(w.min(1.0));
    }
    Range::from_hands_weights(&hands, &weights)
}

fn solve_one(req: &Request) -> Result<Solved, String> {
    let card_config = CardConfig {
        range: [build_range(&req.oop)?, build_range(&req.ip)?],
        flop: flop_from_str(&req.flop)?,
        turn: card_from_str(&req.turn)?,
        river: match &req.river { Some(r) => card_from_str(r)?, None => NOT_DEALT },
    };
    let bets = BetSizeOptions::try_from((req.bets.as_str(), req.raise.as_str()))?;
    let tree_config = TreeConfig {
        initial_state: if req.river.is_some() { BoardState::River } else { BoardState::Turn },
        starting_pot: req.pot,
        effective_stack: req.stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bets.clone(), bets.clone()],
        turn_bet_sizes: [bets.clone(), bets.clone()],
        river_bet_sizes: [bets.clone(), bets],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: req.add_allin,
        force_allin_threshold: req.force_allin,
        merging_threshold: 0.1,
    };
    let mut game = PostFlopGame::with_config(card_config, ActionTree::new(tree_config)?)?;
    game.allocate_memory(false);

    let t0 = Instant::now();
    let target = req.pot as f32 * req.target_pct / 100.0;
    let mut expl = compute_exploitability(&game);
    let mut iters = 0u32;
    while iters < req.max_iter && expl > target {
        solve_step(&game, iters);
        iters += 1;
        if iters % 10 == 0 || iters == req.max_iter {
            expl = compute_exploitability(&game);
        }
    }
    finalize(&mut game);
    let solve_s = t0.elapsed().as_secs_f64();

    game.cache_normalized_weights();
    let (ev0, ev1) = (game.expected_values(0), game.expected_values(1));
    let (w0, w1) = (game.normalized_weights(0).to_vec(), game.normalized_weights(1).to_vec());
    let sum = compute_average(&ev0, &w0) + compute_average(&ev1, &w1);
    Ok(Solved {
        iters,
        expl_pct: 100.0 * expl as f64 / req.pot as f64,
        solve_s,
        ev_sum_err: (sum - req.pot as f32) as f64,
        oop_hands: holes_to_strings(game.private_cards(0))?,
        ip_hands: holes_to_strings(game.private_cards(1))?,
        oop_w: r5s(&w0),
        ip_w: r5s(&w1),
        oop_ev: r5s(&ev0),
        ip_ev: r5s(&ev1),
        root_actions: game.available_actions().iter().map(|a| format!("{a:?}")).collect(),
        root_strategy: r5s(&game.strategy()),
    })
}

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }
        let resp = match serde_json::from_str::<Request>(&line) {
            Err(e) => Response { id: String::new(), ok: false, error: Some(format!("bad request: {e}")), data: None },
            Ok(req) => {
                let id = req.id.clone();
                match catch_unwind(AssertUnwindSafe(|| solve_one(&req))) {
                    Ok(Ok(d)) => Response { id, ok: true, error: None, data: Some(d) },
                    Ok(Err(e)) => Response { id, ok: false, error: Some(e), data: None },
                    Err(_) => Response { id, ok: false, error: Some("solver panicked".into()), data: None },
                }
            }
        };
        writeln!(out, "{}", serde_json::to_string(&resp).unwrap()).unwrap();
        out.flush().unwrap();
    }
}
