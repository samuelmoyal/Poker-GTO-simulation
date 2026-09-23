// Turn-root solve benchmark + label sanity check.
// usage: turn_bench <oop_range_file> <ip_range_file> <flop e.g. Td9d6h> <turn e.g. Qc> <pot> <stack> [target_pct=0.5]
use postflop_solver::*;
use std::{env, fs, time::Instant};

fn main() {
    let a: Vec<String> = env::args().collect();
    let oop = fs::read_to_string(&a[1]).unwrap();
    let ip = fs::read_to_string(&a[2]).unwrap();
    let pot: i32 = a[5].parse().unwrap();
    let stack: i32 = a[6].parse().unwrap();
    let target_pct: f32 = a.get(7).map_or(0.5, |s| s.parse().unwrap());

    let card_config = CardConfig {
        range: [oop.trim().parse().unwrap(), ip.trim().parse().unwrap()],
        flop: flop_from_str(&a[3]).unwrap(),
        turn: card_from_str(&a[4]).unwrap(),
        river: NOT_DEALT,
    };

    // same shape as the trainer's live turn tree: bet 66%, raise 2.5x, all-in
    let bets = BetSizeOptions::try_from(("66%, a", "2.5x")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: pot,
        effective_stack: stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bets.clone(), bets.clone()],
        turn_bet_sizes: [bets.clone(), bets.clone()],
        river_bet_sizes: [bets.clone(), bets],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };

    let t0 = Instant::now();
    let mut game = PostFlopGame::with_config(card_config, ActionTree::new(tree_config).unwrap()).unwrap();
    let (mem, mem_c) = game.memory_usage();
    println!(
        "hands oop/ip: {}/{} | memory {:.2} GB (compressed {:.2} GB) | build {:.2}s",
        game.private_cards(0).len(),
        game.private_cards(1).len(),
        mem as f64 / 1e9,
        mem_c as f64 / 1e9,
        t0.elapsed().as_secs_f32()
    );
    game.allocate_memory(false);

    let t1 = Instant::now();
    let target = pot as f32 * target_pct / 100.0;
    let expl = solve(&mut game, 2000, target, false);
    println!("\nsolve {:.2}s | exploitability {:.3}% of pot (target {target_pct}%)", t1.elapsed().as_secs_f32(), 100.0 * expl / pot as f32);

    let t2 = Instant::now();
    game.cache_normalized_weights();
    let (ev0, ev1) = (game.expected_values(0), game.expected_values(1));
    let (w0, w1) = (game.normalized_weights(0), game.normalized_weights(1));
    let (m0, m1) = (compute_average(&ev0, w0), compute_average(&ev1, w1));
    println!("ev extraction {:.4}s", t2.elapsed().as_secs_f32());
    println!("range-average EV  oop {m0:.3}  ip {m1:.3}  sum {:.4}  (pot {pot})", m0 + m1);
}
