// Flop solve with the trainer's flop tree (flop 50% + raise 100%, turn/river 66% only, no explicit all-in).
// usage: flop_bench <oop_range_file> <ip_range_file> <flop e.g. 7s4h2d> <pot> <stack> [target_pct=0.5] [max_gb=8]
use postflop_solver::*;
use std::{env, fs, time::Instant};

fn main() {
    let a: Vec<String> = env::args().collect();
    let (oop, ip) = (fs::read_to_string(&a[1]).unwrap(), fs::read_to_string(&a[2]).unwrap());
    let (pot, stack): (i32, i32) = (a[4].parse().unwrap(), a[5].parse().unwrap());
    let target_pct: f32 = a.get(6).map_or(0.5, |s| s.parse().unwrap());
    let max_gb: f64 = a.get(7).map_or(8.0, |s| s.parse().unwrap());

    let card_config = CardConfig {
        range: [oop.trim().parse().unwrap(), ip.trim().parse().unwrap()],
        flop: flop_from_str(&a[3]).unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };
    let flop = BetSizeOptions::try_from(("50%", "100%")).unwrap();
    let later = BetSizeOptions::try_from(("66%", "")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: pot,
        effective_stack: stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [flop.clone(), flop],
        turn_bet_sizes: [later.clone(), later.clone()],
        river_bet_sizes: [later.clone(), later],
        turn_donk_sizes: Some(DonkSizeOptions::try_from("66%").unwrap()),
        river_donk_sizes: Some(DonkSizeOptions::try_from("66%").unwrap()),
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };
    let t0 = Instant::now();
    let mut game = PostFlopGame::with_config(card_config, ActionTree::new(tree_config).unwrap()).unwrap();
    let (mem, mem_c) = game.memory_usage();
    println!(
        "hands oop/ip {}/{} | memory {:.2} GB (compressed {:.2} GB) | build {:.1}s",
        game.private_cards(0).len(), game.private_cards(1).len(), mem as f64 / 1e9, mem_c as f64 / 1e9, t0.elapsed().as_secs_f32()
    );
    if mem as f64 / 1e9 > max_gb {
        println!("over {max_gb} GB, aborting");
        return;
    }
    game.allocate_memory(false);
    let t1 = Instant::now();
    let target = pot as f32 * target_pct / 100.0;
    let mut expl = compute_exploitability(&game);
    let mut it = 0u32;
    while it < 1000 && expl > target {
        solve_step(&game, it);
        it += 1;
        if it % 10 == 0 {
            expl = compute_exploitability(&game);
            println!("  it {it:4}  expl {:.3}%  t={:.0}s", 100.0 * expl / pot as f32, t1.elapsed().as_secs_f32());
        }
    }
    finalize(&mut game);
    println!("solve {:.1}s, {it} iterations, exploitability {:.3}% of pot", t1.elapsed().as_secs_f32(), 100.0 * expl / pot as f32);
}
