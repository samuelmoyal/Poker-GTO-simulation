//! Board tables for `net_turn`, in Rust, behind a C ABI (see model/gtonet/fastboard.py).
//!
//! * `score`: 5-7 card hand strength, the SAME integer encoding as gtonet/poker.py (category * 14^5 + 5 base-14 digits),
//!   so results are directly comparable/equal to the numpy evaluator.
//! * `Board` (a 4-card turn board): strengths of all 1326 hands on each of the 48 river cards, sorted once; then
//!   `equity(range)` = conditional equity of every hand against a range, with card removal, by prefix sums over the
//!   sorted order (inclusion-exclusion over the hand's two cards) instead of a 1326x1326 win table:
//!       eq(h) = sum_k [ less_k(h) + tie_k(h)/2 ] / (44 * Z(h)),  Z(h) = range mass not blocked by h.
use rayon::prelude::*;
use std::sync::OnceLock;

const NH: usize = 1326;
const NR: usize = 48; // river cards for a 4-card board
const BASE: i32 = 14;

fn combos() -> &'static [(u8, u8); NH] {
    static C: OnceLock<[(u8, u8); NH]> = OnceLock::new();
    C.get_or_init(|| {
        let mut a = [(0u8, 0u8); NH];
        let mut n = 0;
        for i in 0..52u8 {
            for j in (i + 1)..52u8 {
                a[n] = (i, j);
                n += 1;
            }
        }
        a
    })
}

fn straight_high(mask: u16) -> i32 {
    let mut best = -1;
    if mask & 0b1_0000_0000_1111 == 0b1_0000_0000_1111 {
        best = 3; // the wheel A-2-3-4-5
    }
    for high in 4..=12 {
        let w = 0b11111u16 << (high - 4);
        if mask & w == w {
            best = high as i32;
        }
    }
    best
}

/// top-m ranks present in `mask` not in `excl`, descending, -1 when fewer
fn kick(mask: u16, excl: &[i32], m: usize, out: &mut [i32; 5]) {
    let mut c = 0;
    for r in (0..13).rev() {
        if c == m {
            break;
        }
        if (mask >> r) & 1 == 1 && !excl.contains(&(r as i32)) {
            out[c] = r as i32;
            c += 1;
        }
    }
    for slot in out.iter_mut().take(m).skip(c) {
        *slot = -1;
    }
}

fn pack(cat: i32, cols: &[i32]) -> i32 {
    let mut s = cat;
    for i in 0..5 {
        let d = if i < cols.len() { cols[i] + 1 } else { 0 };
        s = s * BASE + d;
    }
    s
}

/// Strength of 5-7 cards (ids 0..51 = 4 * rank + suit); higher wins, equal = tie.
pub fn score(cards: &[u8]) -> i32 {
    let mut cnt = [0u8; 13];
    let mut suit_cnt = [0u8; 4];
    let mut suit_mask = [0u16; 4];
    let mut present = 0u16;
    for &c in cards {
        let (r, s) = ((c / 4) as usize, (c % 4) as usize);
        cnt[r] += 1;
        suit_cnt[s] += 1;
        suit_mask[s] |= 1 << r;
        present |= 1 << r;
    }
    let mut fs = 0;
    for s in 1..4 {
        if suit_cnt[s] > suit_cnt[fs] {
            fs = s;
        }
    }
    let has_flush = suit_cnt[fs] >= 5;
    let fl = if has_flush { suit_mask[fs] } else { 0 };
    let st = straight_high(present);
    let st_fl = if has_flush { straight_high(fl) } else { -1 };
    let (mut quad, mut t1, mut p1) = (-1, -1, -1);
    for r in 0..13 {
        if cnt[r] == 4 {
            quad = r as i32;
        }
        if cnt[r] >= 3 {
            t1 = r as i32;
        }
        if cnt[r] >= 2 {
            p1 = r as i32;
        }
    }
    let (mut pair_ex_t1, mut p2) = (-1, -1);
    for r in 0..13 {
        if cnt[r] >= 2 {
            if r as i32 != t1 {
                pair_ex_t1 = r as i32;
            }
            if r as i32 != p1 {
                p2 = r as i32;
            }
        }
    }
    let mut k = [-1i32; 5];
    if st_fl >= 0 {
        return pack(8, &[st_fl]);
    }
    if quad >= 0 {
        kick(present, &[quad], 1, &mut k);
        return pack(7, &[quad, k[0]]);
    }
    if t1 >= 0 && pair_ex_t1 >= 0 {
        return pack(6, &[t1, pair_ex_t1]);
    }
    if has_flush {
        kick(fl, &[], 5, &mut k);
        return pack(5, &k);
    }
    if st >= 0 {
        return pack(4, &[st]);
    }
    if t1 >= 0 {
        kick(present, &[t1], 2, &mut k);
        return pack(3, &[t1, k[0], k[1]]);
    }
    if p2 >= 0 {
        kick(present, &[p1, p2], 1, &mut k);
        return pack(2, &[p1, p2, k[0]]);
    }
    if p1 >= 0 {
        kick(present, &[p1], 3, &mut k);
        return pack(1, &[p1, k[0], k[1], k[2]]);
    }
    kick(present, &[], 5, &mut k);
    pack(0, &k)
}

fn disjoint() -> &'static [u8] {
    static D: OnceLock<Vec<u8>> = OnceLock::new();
    D.get_or_init(|| {
        let cs = combos();
        let mut d = vec![0u8; NH * NH];
        for h in 0..NH {
            for g in 0..NH {
                let (a, b, c, e) = (cs[h].0, cs[h].1, cs[g].0, cs[g].1);
                d[h * NH + g] = (a != c && a != e && b != c && b != e) as u8;
            }
        }
        d
    })
}

const CAT_BASE: i32 = 14 * 14 * 14 * 14 * 14;

pub struct Board {
    s7: Vec<i32>,        // [k][h] strengths, -1 = invalid at river k
    pub hand_ok: Vec<bool>,
    pub cat6: Vec<u8>,
    pub outs: Vec<u8>,
    nv: [usize; NR],
    order: Vec<u16>,     // [k][i]  hand ids of the valid hands at river k, ascending strength
    lo: Vec<u16>,        // [k][h]  first position in order[k] of h's strength class
    hi: Vec<u16>,        // [k][h]  one past the last
    sub: Vec<u16>,       // [k][x][j] valid hands containing card x, ascending strength (stride 51)
    sub_n: Vec<u8>,      // [k][x]
    slo: Vec<u8>,        // [k][h][a] class bounds of h inside the sub-list of its a-th card
    shi: Vec<u8>,
}

const STRIDE: usize = 51;

impl Board {
    pub fn new(board: [u8; 4]) -> Board {
        let cs = combos();
        let hand_ok: Vec<bool> = cs.iter().map(|&(a, b)| !board.contains(&a) && !board.contains(&b)).collect();
        let rivers: Vec<u8> = (0..52u8).filter(|c| !board.contains(c)).collect();
        assert_eq!(rivers.len(), NR);
        let mut by_card: Vec<Vec<u16>> = vec![Vec::new(); 52];
        for (h, &(a, b)) in cs.iter().enumerate() {
            by_card[a as usize].push(h as u16);
            by_card[b as usize].push(h as u16);
        }
        let mut cat6 = vec![0u8; NH];
        let mut cat6_full = vec![0i32; NH];
        for h in 0..NH {
            if hand_ok[h] {
                let (a, b) = cs[h];
                cat6_full[h] = score(&[a, b, board[0], board[1], board[2], board[3]]) / CAT_BASE;
                cat6[h] = cat6_full[h] as u8;
            }
        }
        let mut outs = vec![0u8; NH];
        let mut order = vec![0u16; NR * NH];
        let mut lo = vec![0u16; NR * NH];
        let mut hi = vec![0u16; NR * NH];
        let mut sub = vec![0u16; NR * 52 * STRIDE];
        let mut sub_n = vec![0u8; NR * 52];
        let mut slo = vec![0u8; NR * NH * 2];
        let mut shi = vec![0u8; NR * NH * 2];
        let mut nv = [0usize; NR];
        let mut s = vec![-1i32; NH];
        let mut s7 = vec![-1i32; NR * NH];
        for (k, &c) in rivers.iter().enumerate() {
            for h in 0..NH {
                let (a, b) = cs[h];
                s[h] = if hand_ok[h] && a != c && b != c { score(&[a, b, board[0], board[1], board[2], board[3], c]) } else { -1 };
                if s[h] >= 0 && s[h] / CAT_BASE > cat6_full[h] {
                    outs[h] += 1;
                }
            }
            s7[k * NH..(k + 1) * NH].copy_from_slice(&s);
            let mut valid: Vec<u16> = (0..NH as u16).filter(|&h| s[h as usize] >= 0).collect();
            valid.sort_by_key(|&h| (s[h as usize], h));
            nv[k] = valid.len();
            let scores: Vec<i32> = valid.iter().map(|&h| s[h as usize]).collect();
            for (i, &h) in valid.iter().enumerate() {
                order[k * NH + i] = h;
                let sc = s[h as usize];
                lo[k * NH + h as usize] = scores.partition_point(|&x| x < sc) as u16;
                hi[k * NH + h as usize] = scores.partition_point(|&x| x <= sc) as u16;
            }
            for x in 0..52 {
                let mut l: Vec<u16> = by_card[x].iter().copied().filter(|&h| s[h as usize] >= 0).collect();
                l.sort_by_key(|&h| (s[h as usize], h));
                sub_n[k * 52 + x] = l.len() as u8;
                let ls: Vec<i32> = l.iter().map(|&h| s[h as usize]).collect();
                for (j, &h) in l.iter().enumerate() {
                    sub[(k * 52 + x) * STRIDE + j] = h;
                }
                for &h in &l {
                    let (a, b) = cs[h as usize];
                    let side = if a as usize == x { 0 } else { 1 };
                    debug_assert!(a as usize == x || b as usize == x);
                    let sc = s[h as usize];
                    slo[(k * NH + h as usize) * 2 + side] = ls.partition_point(|&v| v < sc) as u8;
                    shi[(k * NH + h as usize) * 2 + side] = ls.partition_point(|&v| v <= sc) as u8;
                }
            }
        }
        Board { s7, hand_ok, cat6, outs, nv, order, lo, hi, sub, sub_n, slo, shi }
    }

    /// WD[h][h'] = D(h,h') * mean over river cards of outcome(h vs h') (win 1, tie 1/2), 0.5 for pairs a board-blocked
    /// hand is in, 0 for hands sharing a card: the matrix `equity(r) = (WD r) / (D r)` of gtonet/equity.py.
    pub fn win_table(&self, out: &mut [f32]) {
        let disj = disjoint();
        let mut acc = vec![0u8; NH * NH];
        for k in 0..NR {
            let sk = &self.s7[k * NH..(k + 1) * NH];
            for h in 0..NH {
                let sh = sk[h];
                if sh < 0 {
                    continue;
                }
                let arow = &mut acc[h * NH..(h + 1) * NH];
                let drow = &disj[h * NH..(h + 1) * NH];
                for ((a, &sp), &d) in arow.iter_mut().zip(sk.iter()).zip(drow.iter()) {
                    let m = ((sp >= 0) as u8) & d;
                    *a += ((((sh > sp) as u8) << 1) | ((sh == sp) as u8)) * m;   // win = 2 units, tie = 1
                }
            }
        }
        for h in 0..NH {
            for hp in 0..NH {
                out[h * NH + hp] = if disj[h * NH + hp] == 0 {
                    0.0
                } else if self.hand_ok[h] && self.hand_ok[hp] {
                    acc[h * NH + hp] as f32 / 88.0                          // 44 shared river cards x 2 units
                } else {
                    0.5
                };
            }
        }
    }

    /// conditional equity of every hand vs the range `r` (any scale), 0.5 where undefined; out has NH entries
    pub fn equity(&self, r: &[f32], out: &mut [f32]) {
        let cs = combos();
        let rr: Vec<f64> = (0..NH).map(|h| if self.hand_ok[h] { r[h] as f64 } else { 0.0 }).collect();
        let total: f64 = rr.iter().sum();
        let mut m = [0f64; 52];
        for h in 0..NH {
            let (a, b) = cs[h];
            m[a as usize] += rr[h];
            m[b as usize] += rr[h];
        }
        let mut num = vec![0f64; NH];
        let mut cum_a = vec![0f64; NH + 1];
        let mut cb = vec![0f64; 52 * (STRIDE + 1)];
        for k in 0..NR {
            let nv = self.nv[k];
            for i in 0..nv {
                cum_a[i + 1] = cum_a[i] + rr[self.order[k * NH + i] as usize];
            }
            for x in 0..52 {
                let n = self.sub_n[k * 52 + x] as usize;
                let base = x * (STRIDE + 1);
                cb[base] = 0.0;
                for j in 0..n {
                    cb[base + j + 1] = cb[base + j] + rr[self.sub[(k * 52 + x) * STRIDE + j] as usize];
                }
            }
            for i in 0..nv {
                let h = self.order[k * NH + i] as usize;
                let (c1, c2) = (cs[h].0 as usize * (STRIDE + 1), cs[h].1 as usize * (STRIDE + 1));
                let (l, hh) = (self.lo[k * NH + h] as usize, self.hi[k * NH + h] as usize);
                let (l1, h1) = (self.slo[(k * NH + h) * 2] as usize, self.shi[(k * NH + h) * 2] as usize);
                let (l2, h2) = (self.slo[(k * NH + h) * 2 + 1] as usize, self.shi[(k * NH + h) * 2 + 1] as usize);
                let less = cum_a[l] - cb[c1 + l1] - cb[c2 + l2];
                let tie = (cum_a[hh] - cum_a[l]) - (cb[c1 + h1] - cb[c1 + l1]) - (cb[c2 + h2] - cb[c2 + l2]) + rr[h];
                num[h] += less + 0.5 * tie;
            }
        }
        for h in 0..NH {
            let (a, b) = cs[h];
            let z = total - m[a as usize] - m[b as usize] + rr[h];
            out[h] = if self.hand_ok[h] && z > 1e-12 { (num[h] / (44.0 * z)) as f32 } else { 0.5 };
        }
    }
}

// ------------------------------------------------------------------------------------------------ C ABI
#[no_mangle]
pub extern "C" fn gt_version() -> i32 {
    1
}

/// n hands of k cards each (row-major u8) -> strengths
#[no_mangle]
pub unsafe extern "C" fn gt_strength(cards: *const u8, n: usize, k: usize, out: *mut i32) {
    let cards = std::slice::from_raw_parts(cards, n * k);
    let out = std::slice::from_raw_parts_mut(out, n);
    out.par_iter_mut().enumerate().for_each(|(i, o)| *o = score(&cards[i * k..(i + 1) * k]));
}

/// n boards (4 u8 each) -> n heap boards written to `out`
#[no_mangle]
pub unsafe extern "C" fn gt_boards_new(boards: *const u8, n: usize, out: *mut *mut Board) {
    let b = std::slice::from_raw_parts(boards, n * 4);
    let out = std::slice::from_raw_parts_mut(out, n);
    let made: Vec<Board> = (0..n).into_par_iter().map(|i| Board::new([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])).collect();
    for (o, m) in out.iter_mut().zip(made) {
        *o = Box::into_raw(Box::new(m));
    }
}

#[no_mangle]
pub unsafe extern "C" fn gt_board_free(p: *mut Board) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

#[no_mangle]
pub unsafe extern "C" fn gt_board_info(p: *const Board, cat6: *mut u8, outs: *mut u8, hand_ok: *mut u8) {
    let b = &*p;
    std::slice::from_raw_parts_mut(cat6, NH).copy_from_slice(&b.cat6);
    std::slice::from_raw_parts_mut(outs, NH).copy_from_slice(&b.outs);
    for (o, &h) in std::slice::from_raw_parts_mut(hand_ok, NH).iter_mut().zip(&b.hand_ok) {
        *o = h as u8;
    }
}

/// for board j (of nb) evaluate m ranges r[j][i][1326] -> out[j][i][1326]
#[no_mangle]
pub unsafe extern "C" fn gt_equity_multi(boards: *const *const Board, nb: usize, r: *const f32, m: usize, out: *mut f32) {
    let boards: Vec<&Board> = std::slice::from_raw_parts(boards, nb).iter().map(|&p| &*p).collect();
    let r = std::slice::from_raw_parts(r, nb * m * NH);
    let out = std::slice::from_raw_parts_mut(out, nb * m * NH);
    out.par_chunks_mut(NH).enumerate().for_each(|(idx, o)| {
        let j = idx / m;
        boards[j].equity(&r[idx * NH..(idx + 1) * NH], o);
    });
}

/// n boards -> their (1326 x 1326) float32 tables, board-major, in parallel
#[no_mangle]
pub unsafe extern "C" fn gt_boards_wd(boards: *const *const Board, n: usize, out: *mut f32) {
    let boards: Vec<&Board> = std::slice::from_raw_parts(boards, n).iter().map(|&p| &*p).collect();
    let out = std::slice::from_raw_parts_mut(out, n * NH * NH);
    out.par_chunks_mut(NH * NH).enumerate().for_each(|(j, o)| boards[j].win_table(o));
}
