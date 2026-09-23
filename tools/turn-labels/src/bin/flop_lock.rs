//! Solve a flop exactly, optionally with some flop strategies LOCKED, and report each player's root EV.
//!
//! Real EV loss of a flop strategy S of player p:  loss_p = V_p(NE) - V_p(S locked at the flop, everything else optimal).
//! (locks: {"path":"x-b","player":1,"strategy":{"AhKh":{"f":0.2,"c":0.8}}}; path = action TYPES from the root:
//!  x check, b bet, r raise, c call, f fold, a all-in; player 0 = OOP, 1 = IP.)
//!
//! request : {"id","flop","pot","stack","oop":{hand:w},"ip":{..},"target_pct","max_iter","max_gb","probe",
//!            "flop_bets","flop_raise","later_bets","later_raise","donk","add_allin","force_allin","locks":[..]}
//! response: {"id","ok","mem_gb","iters","expl_pct","solve_s","ev_oop","ev_ip","nodes":{path:{player,actions,hands,strategy}}}
use postflop_solver::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

fn d_target() -> f32 { 0.05 }
fn d_iters() -> u32 { 1000 }
fn d_gb() -> f64 { 6.0 }
fn d_s(x: &str) -> String { x.to_string() }
fn d_flop_bets() -> String { d_s("50%") }
fn d_later() -> String { d_s("66%") }
fn d_force() -> f64 { 0.15 }

#[derive(Deserialize)]
struct Lock {
    path: String,
    player: usize,
    strategy: HashMap<String, HashMap<String, f32>>,
}

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
    #[serde(default)] probe: bool,
    #[serde(default = "d_flop_bets")] flop_bets: String,
    #[serde(default)] flop_raise: String,
    #[serde(default = "d_later")] later_bets: String,
    #[serde(default)] later_raise: String,
    #[serde(default = "d_later")] donk: String,
    #[serde(default)] add_allin: f64,
    #[serde(default = "d_force")] force_allin: f64,
    #[serde(default)] locks: Vec<Lock>,
}

#[derive(Serialize)]
struct NodeOut {
    player: usize,
    actions: Vec<String>,
    hands: Vec<String>,
    strategy: Vec<f64>,
}

#[derive(Serialize, Default)]
struct Solved {
    mem_gb: f64,
    iters: u32,
    expl_pct: f64,
    solve_s: f64,
    ev_oop: f64,
    ev_ip: f64,
    nodes: HashMap<String, NodeOut>,
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

fn kind(a: &Action) -> String {
    let s = format!("{a:?}");
    if s.starts_with("Check") { "x" } else if s.starts_with("Call") { "c" } else if s.starts_with("Fold") { "f" }
    else if s.starts_with("Bet") { "b" } else if s.starts_with("Raise") { "r" } else if s.starts_with("AllIn") { "a" } else { "?" }
        .to_string()
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

fn walk_to(game: &mut PostFlopGame, path: &str) -> Result<(), String> {
    game.back_to_root();
    for tok in path.split('-').filter(|t| !t.is_empty()) {
        let idx = game.available_actions().iter().position(|a| kind(a) == tok).ok_or(format!("no action {tok} in {path}"))?;
        game.play(idx);
    }
    Ok(())
}

fn apply_lock(game: &mut PostFlopGame, lk: &Lock) -> Result<(), String> {
    walk_to(game, &lk.path)?;
    if game.is_terminal_node() || game.is_chance_node() || game.current_player() != lk.player {
        return Err(format!("lock at {:?}: not a decision node of player {}", lk.path, lk.player));
    }
    let types: Vec<String> = game.available_actions().iter().map(kind).collect();
    let hands: Vec<(u8, u8)> = game.private_cards(lk.player).to_vec();
    let n = hands.len();
    let mut by_pair: HashMap<(u8, u8), &HashMap<String, f32>> = HashMap::new();
    for (h, m) in &lk.strategy {
        let (a, b) = (card_from_str(&h[0..2])?, card_from_str(&h[2..4])?);
        by_pair.insert((a.min(b), a.max(b)), m);
    }
    let mut v = vec![0f32; types.len() * n];
    for (j, &(c1, c2)) in hands.iter().enumerate() {
        if let Some(m) = by_pair.get(&(c1.min(c2), c1.max(c2))) {
            let probs: Vec<f32> = types.iter().map(|t| *m.get(t).unwrap_or(&0.0)).collect();
            if probs.iter().sum::<f32>() > 0.0 {
                for (a, p) in probs.iter().enumerate() { v[a * n + j] = *p; }
            }
        }
    }
    game.lock_current_strategy(&v);
    Ok(())
}

fn collect(game: &mut PostFlopGame, hist: &mut Vec<usize>, path: &mut Vec<String>, out: &mut HashMap<String, NodeOut>) {
    game.apply_history(hist);
    if game.is_terminal_node() || game.is_chance_node() { return; }
    let acts = game.available_actions();
    let types: Vec<String> = acts.iter().map(kind).collect();
    let p = game.current_player();
    out.insert(path.join("-"), NodeOut {
        player: p, actions: types.clone(),
        hands: holes_to_strings(game.private_cards(p)).unwrap(),
        strategy: game.strategy().iter().map(|&x| x as f64).collect(),
    });
    for i in 0..acts.len() {
        if types[i] == "f" { continue; }
        hist.push(i);
        path.push(types[i].clone());
        collect(game, hist, path, out);
        hist.pop();
        path.pop();
    }
}

fn solve_one(req: &Request) -> Result<Solved, String> {
    let card_config = CardConfig { range: [build_range(&req.oop)?, build_range(&req.ip)?], flop: flop_from_str(&req.flop)?, turn: NOT_DEALT, river: NOT_DEALT };
    let flop = BetSizeOptions::try_from((req.flop_bets.as_str(), req.flop_raise.as_str()))?;
    let later = BetSizeOptions::try_from((req.later_bets.as_str(), req.later_raise.as_str()))?;
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop, starting_pot: req.pot, effective_stack: req.stack, rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [flop.clone(), flop], turn_bet_sizes: [later.clone(), later.clone()], river_bet_sizes: [later.clone(), later],
        turn_donk_sizes: if req.donk.is_empty() { None } else { Some(DonkSizeOptions::try_from(req.donk.as_str())?) },
        river_donk_sizes: if req.donk.is_empty() { None } else { Some(DonkSizeOptions::try_from(req.donk.as_str())?) },
        add_allin_threshold: req.add_allin, force_allin_threshold: req.force_allin, merging_threshold: 0.1,
    };
    let mut game = PostFlopGame::with_config(card_config, ActionTree::new(tree_config)?)?;
    let mem_gb = game.memory_usage().0 as f64 / 1e9;
    if req.probe || mem_gb > req.max_gb {
        return if req.probe { Ok(Solved { mem_gb, ..Default::default() }) } else { Err(format!("tree needs {mem_gb:.1} GB > limit {}", req.max_gb)) };
    }
    game.allocate_memory(false);
    for lk in &req.locks { apply_lock(&mut game, lk)?; }
    game.back_to_root();

    let t0 = Instant::now();
    let target = req.pot as f32 * req.target_pct / 100.0;
    let mut expl = compute_exploitability(&game);
    let mut iters = 0u32;
    while iters < req.max_iter && expl > target {
        solve_step(&game, iters);
        iters += 1;
        if iters % 10 == 0 || iters == req.max_iter { expl = compute_exploitability(&game); }
    }
    finalize(&mut game);
    let solve_s = t0.elapsed().as_secs_f64();

    game.back_to_root();
    game.cache_normalized_weights();
    let (ev0, ev1) = (game.expected_values(0), game.expected_values(1));
    let (w0, w1) = (game.normalized_weights(0).to_vec(), game.normalized_weights(1).to_vec());
    let (ev_oop, ev_ip) = (compute_average(&ev0, &w0) as f64, compute_average(&ev1, &w1) as f64);
    let mut nodes = HashMap::new();
    collect(&mut game, &mut Vec::new(), &mut Vec::new(), &mut nodes);
    Ok(Solved { mem_gb, iters, expl_pct: 100.0 * expl as f64 / req.pot as f64, solve_s, ev_oop, ev_ip, nodes })
}

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
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
