//! JSONL in (stdin) -> JSONL out (stdout): solve a flop, then list every way the flop betting can end in a
//! turn card, with both players' per-combo ranges at that point (reach = preflop weight x own action probs).
//!
//! request : {"id","flop":"7s4h2d","pot":55,"stack":975,"oop":{"AhKs":0.4,..},"ip":{..},"target_pct":0.5,
//!            "max_iter":1000,"max_gb":6,"flop_bets":"50%","flop_raise":"100%","later_bets":"66%","donk":"66%"}
//! response: {"id","ok":true,"iters","expl_pct","solve_s","mem_gb",
//!            "lines":[{"line":"x-b28-c","pot","stack","oop":{hand:reach},"ip":{..}}]}
//! Line labels: x check, c call, b<N> bet, r<N> raise-to, a<N> all-in (chips; folds end a line and are skipped).
use postflop_solver::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

fn d_target() -> f32 { 0.5 }
fn d_iters() -> u32 { 1000 }
fn d_gb() -> f64 { 6.0 }
fn d_flop_bets() -> String { "50%".into() }
fn d_flop_raise() -> String { "100%".into() }
fn d_later() -> String { "66%".into() }

#[derive(Deserialize)]
struct Request {
    id: String,
    flop: String,
    pot: i32,
    stack: i32,
    oop: HashMap<String, f32>,
    ip: HashMap<String, f32>,
    #[serde(default = "d_target")] target_pct: f32,
    #[serde(default = "d_iters")] max_iter: u32,
    #[serde(default = "d_gb")] max_gb: f64,
    #[serde(default = "d_flop_bets")] flop_bets: String,
    #[serde(default = "d_flop_raise")] flop_raise: String,
    #[serde(default = "d_later")] later_bets: String,
    #[serde(default = "d_later")] donk: String,
}

#[derive(Serialize)]
struct Line {
    line: String,
    pot: i32,
    stack: i32,
    oop: HashMap<String, f64>,
    ip: HashMap<String, f64>,
}

#[derive(Serialize, Default)]
struct Solved {
    iters: u32,
    expl_pct: f64,
    solve_s: f64,
    mem_gb: f64,
    lines: Vec<Line>,
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

fn build_range(m: &HashMap<String, f32>) -> Result<Range, String> {
    let (mut hands, mut weights) = (Vec::new(), Vec::new());
    for (h, &w) in m {
        if w <= 0.0 || h.len() != 4 { continue; }
        hands.push((card_from_str(&h[0..2])?, card_from_str(&h[2..4])?));
        weights.push(w.min(1.0));
    }
    Range::from_hands_weights(&hands, &weights)
}

fn label(action: &Action) -> Option<String> {
    let s = format!("{action:?}");
    match s.as_str() {
        "Fold" => None,
        "Check" => Some("x".into()),
        "Call" => Some("c".into()),
        _ => {
            let n = s.split('(').nth(1)?.trim_end_matches(')').to_string();
            let tag = if s.starts_with("Bet") { "b" } else if s.starts_with("Raise") { "r" } else if s.starts_with("AllIn") { "a" } else { return None };
            Some(format!("{tag}{n}"))
        }
    }
}

fn weights_map(hands: &[String], w: &[f32]) -> HashMap<String, f64> {
    hands.iter().zip(w).filter(|(_, &x)| x > 0.0).map(|(h, &x)| (h.clone(), ((x as f64) * 1e6).round() / 1e6)).collect()
}

fn walk(game: &mut PostFlopGame, hist: &mut Vec<usize>, line: &mut Vec<String>, hands: &[Vec<String>; 2], pot0: i32, stack0: i32, out: &mut Vec<Line>) {
    game.apply_history(hist);
    if game.is_terminal_node() { return; }
    if game.is_chance_node() {
        let tb = game.total_bet_amount();
        out.push(Line {
            line: line.join("-"),
            pot: pot0 + tb[0] + tb[1],
            stack: stack0 - tb[0].max(tb[1]),
            oop: weights_map(&hands[0], game.weights(0)),
            ip: weights_map(&hands[1], game.weights(1)),
        });
        return;
    }
    for (i, a) in game.available_actions().iter().enumerate() {
        if let Some(l) = label(a) {
            hist.push(i);
            line.push(l);
            walk(game, hist, line, hands, pot0, stack0, out);
            hist.pop();
            line.pop();
        }
    }
}

fn solve_one(req: &Request) -> Result<Solved, String> {
    let card_config = CardConfig {
        range: [build_range(&req.oop)?, build_range(&req.ip)?],
        flop: flop_from_str(&req.flop)?,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };
    let flop = BetSizeOptions::try_from((req.flop_bets.as_str(), req.flop_raise.as_str()))?;
    let later = BetSizeOptions::try_from((req.later_bets.as_str(), ""))?;
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: req.pot,
        effective_stack: req.stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [flop.clone(), flop],
        turn_bet_sizes: [later.clone(), later.clone()],
        river_bet_sizes: [later.clone(), later],
        turn_donk_sizes: Some(DonkSizeOptions::try_from(req.donk.as_str())?),
        river_donk_sizes: Some(DonkSizeOptions::try_from(req.donk.as_str())?),
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };
    let mut game = PostFlopGame::with_config(card_config, ActionTree::new(tree_config)?)?;
    let mem_gb = game.memory_usage().0 as f64 / 1e9;
    if mem_gb > req.max_gb {
        return Err(format!("tree needs {mem_gb:.1} GB > limit {}", req.max_gb));
    }
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

    let hands = [holes_to_strings(game.private_cards(0))?, holes_to_strings(game.private_cards(1))?];
    let mut lines = Vec::new();
    walk(&mut game, &mut Vec::new(), &mut Vec::new(), &hands, req.pot, req.stack, &mut lines);
    Ok(Solved { iters, expl_pct: 100.0 * expl as f64 / req.pot as f64, solve_s, mem_gb, lines })
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
