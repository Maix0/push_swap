#![allow(dead_code)]
#![allow(unused)]
use std::collections::{vec_deque::VecDeque, HashSet};

use itertools::Itertools;
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use statrs::statistics::{Distribution, *};

#[derive(Debug, Clone, Eq, PartialEq, Hash, Copy)]
enum Move {
    RotateA,
    RotateB,
    RotateBoth,
    RevRotateA,
    RevRotateB,
    RevRotateBoth,
    PushA,
    PushB,
}

type Stack<T = i32> = VecDeque<T>;

#[derive(Clone)]
struct State {
    a: Stack,
    b: Stack,
    sorted: Stack<(i32, bool)>,
    counts: usize,
    moves: Vec<Move>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Copy)]
pub enum Rotates {
    Forward(usize),
    Reverse(usize),
}

impl std::cmp::PartialOrd for Rotates {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for Rotates {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let (Self::Forward(lhs) | Self::Reverse(lhs)) = self;
        let (Self::Forward(rhs) | Self::Reverse(rhs)) = other;
        lhs.cmp(rhs)
    }
}

impl Rotates {
    fn flip(self, n: usize) -> Self {
        match self {
            Self::Reverse(r) => Self::Forward(n - r),
            Self::Forward(f) => Self::Reverse(n - f),
        }
    }
}

fn find_place(elem: i32, state: &State) -> usize {
    state
        .sorted
        .iter()
        .copied()
        .filter(|&(e, active)| active || elem == e)
        .enumerate()
        .find(|(_, (e, _))| *e == elem)
        .map(|(i, _)| i)
        .unwrap_or_else(|| {
            println!("why...");
            0
        })
}

fn target(from: usize, to: usize, stack_len: usize) -> Rotates {
    let i = from as usize;
    let j = to as usize;
    let n = stack_len;
    // min(|i−j|,n−|i−j|)
    //Rotates::Forward((i - j).unsigned_abs()).min(Rotates::Reverse(n - (i - j).unsigned_abs()))
    if i < j {
        Rotates::Reverse(j - i)
    } else {
        Rotates::Forward(i - j)
    }
}

enum StackSelector {
    A,
    B,
}

impl Rotates {
    fn action(&self, state: &mut State, stack: StackSelector) {
        let mut func = match self {
            Self::Forward(_) => match stack {
                StackSelector::A => ra,
                StackSelector::B => rb,
            },
            Self::Reverse(_) => match stack {
                StackSelector::A => rra,
                StackSelector::B => rrb,
            },
        };
        let (Self::Forward(n) | Self::Reverse(n)) = *self;
        for _ in 0..n {
            (func)(state);
        }
    }
}

fn optimize_moves(state: &mut State, mut cpy: State) -> usize {
    let mut start_index = 0;
    let mut current_len = 0;
    let mut optimized = Vec::new();
    //println!("{:?}", &state.moves);
    optimized = state.moves.clone();
    /*while start_index + current_len < state.moves.len() {
        match state.moves[start_index + current_len] {
            Move::RotateA
            | Move::RotateB
            | Move::RotateBoth
            | Move::RevRotateA
            | Move::RevRotateB
            | Move::RevRotateBoth => {
                current_len += 1;
            }
            m @ (Move::PushB | Move::PushA) => {
                if current_len != 0 {
                    println!("{:?}", &state.moves[start_index..][..current_len]);
                    optimize_moves_span(&state.moves[start_index..][..current_len], &mut optimized);
                    start_index += current_len;
                    current_len = 0;
                }
                start_index += 1;
                optimized.push(m);
            }
        }
    }
    if current_len != 0 {
        optimize_moves_span(&state.moves[start_index..], &mut optimized);
    }*/
    for m in &state.moves {
        match *m {
            Move::PushA => pa(&mut cpy),
            Move::PushB => pb(&mut cpy),
            Move::RotateA => ra(&mut cpy),
            Move::RotateB => rb(&mut cpy),
            Move::RotateBoth => {
                ra(&mut cpy);
                rb(&mut cpy);
            }
            Move::RevRotateA => rra(&mut cpy),
            Move::RevRotateB => rrb(&mut cpy),
            Move::RevRotateBoth => {
                rra(&mut cpy);
                rrb(&mut cpy);
            }
        }
    }
    if is_sorted(cpy.a.iter()) {
        cpy.counts
    } else {
        //println!("failed...");
        //println!("{:?}", cpy.a);
        0
    }
}

fn optimize_moves_span(mut span: &[Move], output: &mut Vec<Move>) {
    while !span.is_empty() {
        let first = span.get(0).unwrap();
        let position = span
            .iter()
            .find_position(|s| *s != first)
            .map(|v| (v.0, *v.1))
            .unwrap_or_else(|| (span.len() - 1, *first));
        match (*first, position.1) {
            (Move::RotateA, Move::RotateB)
            | (Move::RotateB, Move::RotateA)
            | (Move::RevRotateA, Move::RevRotateB)
            | (Move::RevRotateB, Move::RevRotateA) => {
                let (to_process, next) = span.split_at(position.0 + 1);
                span = next;
                optimize_moves_span_inner(to_process, output);
                println!("POMME");
                //println!("{to_process:?}");
            }
            _ => {
                let (to_process, next) = span.split_at(position.0 + 1);
                span = next;
                output.extend_from_slice(to_process);
            }
        }
    }
}

fn optimize_moves_span_inner(span: &[Move], output: &mut Vec<Move>) {
    let ra = span.iter().filter(|m| matches!(m, Move::RotateA)).count();
    let rra = span
        .iter()
        .filter(|m| matches!(m, Move::RevRotateA))
        .count();
    let rb = span.iter().filter(|m| matches!(m, Move::RotateB)).count();
    let rrb = span
        .iter()
        .filter(|m| matches!(m, Move::RevRotateB))
        .count();

    match ((ra, rb), (rra, rrb)) {
        ((0, 0), (a, b)) => {
            if a > b {
                for _ in 0..b {
                    output.push(Move::RotateBoth);
                }
                for _ in b..a {
                    output.push(Move::RotateA);
                }
            } else {
                for _ in 0..a {
                    output.push(Move::RotateBoth);
                }
                for _ in a..b {
                    output.push(Move::RotateB);
                }
            }
        }
        ((a, b), (0, 0)) => {
            if a > b {
                for _ in 0..b {
                    output.push(Move::RevRotateBoth);
                }
                for _ in b..a {
                    output.push(Move::RevRotateA);
                }
            } else {
                for _ in 0..a {
                    output.push(Move::RevRotateBoth);
                }
                for _ in a..b {
                    output.push(Move::RevRotateB);
                }
            }
        }
        _ => {
            output.extend_from_slice(span);
        }
    }
}

fn do_move(state: &mut State, index: usize) {
    let target_index = (find_place(state.a[index], state)
        + (state.b.len() - state.b.iter().position_max().unwrap_or(0)))
        % state.b.len();
    let mut rotate_a = target(0, index, state.a.len());
    let mut rotate_b = target(target_index, 0, state.b.len());

    if (std::mem::discriminant(&rotate_a) != std::mem::discriminant(&rotate_b)) {
        let diff_flip_a = {
            let (Rotates::Forward(a) | Rotates::Reverse(a)) = rotate_a.flip(state.a.len());
            let (Rotates::Forward(b) | Rotates::Reverse(b)) = rotate_b; //.flip(state.a.len());
            a.abs_diff(b)
        };
        let diff_flip_b = {
            let (Rotates::Forward(a) | Rotates::Reverse(a)) = rotate_a; // .flip(state.a.len());
            let (Rotates::Forward(b) | Rotates::Reverse(b)) = rotate_b.flip(state.b.len());
            a.abs_diff(b)
        };
        if (diff_flip_a > diff_flip_b) {
            rotate_b = rotate_b.flip(state.b.len());
        } else {
            rotate_a = rotate_a.flip(state.a.len());
        }
    }

    let (Rotates::Forward(a) | Rotates::Reverse(a)) = rotate_a; // .flip(state.a.len());
    let (Rotates::Forward(b) | Rotates::Reverse(b)) = rotate_b;
    for _ in 0..a.min(b) {
        match rotate_a {
            Rotates::Forward(_) => rr(state),
            Rotates::Reverse(_) => rrr(state),
        }
    }
    if a < b {
        for _ in 0..(b - a) {
            match rotate_a {
                Rotates::Forward(_) => rb(state),
                Rotates::Reverse(_) => rrb(state),
            }
        }
    } else {
        for _ in 0..(a - b) {
            match rotate_a {
                Rotates::Forward(_) => ra(state),
                Rotates::Reverse(_) => rra(state),
            }
        }
    }
    pb(state);
    if let Some(e) = state.sorted.iter_mut().find(|(e, _)| *e == state.b[0]) {
        e.1 = true;
    }

    //target(
    //    0,
    //    state.b.iter().position_min().unwrap_or_default(),
    //    state.b.len(),
    //)
    //.action(state, StackSelector::B)
}

const PRINT_ACTIONS: bool = false;

macro_rules! print_action {
    ($state:expr, $($t:tt)*) => {
        if PRINT_ACTIONS {
            let writer: &mut dyn std::io::Write = &mut std::io::stdout();
            write!(writer, $($t)*).unwrap();
        }
    };
}

fn is_sorted<I>(data: I) -> bool
where
    I: IntoIterator,
    I::Item: Ord + Clone,
{
    data.into_iter().tuple_windows().all(|(a, b)| a <= b)
}

fn ra(state: &mut State) {
    state.a.rotate_right(1);
    print_action!(state, "RA");
    state.counts += 1;
    state.moves.push(Move::RotateA)
}

fn rb(state: &mut State) {
    state.b.rotate_right(1);
    print_action!(state, "RB");
    state.counts += 1;
    state.moves.push(Move::RotateA)
}

fn rr(state: &mut State) {
    state.a.rotate_right(1);
    state.b.rotate_right(1);
    print_action!(state, "RR");
    state.counts += 1;
    state.moves.push(Move::RotateBoth)
}

fn rra(state: &mut State) {
    state.a.rotate_left(1);
    print_action!(state, "RRA");
    state.counts += 1;
    state.moves.push(Move::RevRotateA)
}

fn rrb(state: &mut State) {
    state.b.rotate_left(1);
    print_action!(state, "RRB");
    state.counts += 1;
    state.moves.push(Move::RevRotateB)
}

fn rrr(state: &mut State) {
    state.a.rotate_left(1);
    state.b.rotate_left(1);
    print_action!(state, "RRR");
    state.counts += 1;
    state.moves.push(Move::RevRotateBoth)
}

fn pa(state: &mut State) {
    if let Some(e) = state.b.pop_front() {
        state.a.push_front(e);
    }
    print_action!(state, "PA");
    state.counts += 1;
    state.moves.push(Move::PushA)
}

fn pb(state: &mut State) {
    if let Some(e) = state.a.pop_front() {
        state.b.push_front(e);
    }
    print_action!(state, "PB");
    state.counts += 1;
    state.moves.push(Move::PushB)
}

fn main() {
    let iter_size = std::env::args()
        .nth(1)
        .map(|i| i.parse::<u32>().unwrap_or(64))
        .unwrap_or(64)
        .max(2);
    let iter_numbers = std::env::args()
        .nth(2)
        .map(|i| i.parse::<u32>().unwrap_or(1024))
        .unwrap_or(1024)
        .max(1);

    let results = (0..iter_numbers)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let mut items = HashSet::with_capacity(iter_size as usize);
            while items.len() < iter_size as usize {
                let n = rng.gen::<i32>();
                if items.contains(&n) {
                    continue;
                }
                items.insert(n);
            }
            run_with_items(items.into_iter())
        })
        .collect::<Vec<_>>();
    if results.iter().any(Result::is_err) {
        println!("There has been a sequence that didn't sort correctly !");
    } else {
        let data = Data::new(
            results
                .iter()
                .copied()
                .map(Result::unwrap)
                .map(|i| (i.0) as f64)
                .collect::<Vec<_>>(),
        );
        println!("Ran {iter_numbers} test with {iter_size} sized inputs");
        println!("========================================");
        println!("Mean   \t=> {}", data.mean().unwrap());
        println!("Median \t=> {}", data.median());
        println!("StdDev \t=> {}", data.std_dev().unwrap());
        println!("Min    \t=> {}", data.min());
        println!("Max    \t=> {}", data.max());
        return;
        let data = Data::new(
            results
                .iter()
                .copied()
                .map(Result::unwrap)
                .map(|i| (i.1) as f64)
                .collect::<Vec<_>>(),
        );
        println!("Ran {iter_numbers} test with {iter_size} sized inputs");
        println!("========================================");
        println!("Mean   \t=> {}", data.mean().unwrap());
        println!("Median \t=> {}", data.median());
        println!("StdDev \t=> {}", data.std_dev().unwrap());
        println!("Min    \t=> {}", data.min());
        println!("Max    \t=> {}", data.max());
    }
}

fn sort_three_b(state: &mut State) {
    let mut c = state.b.clone();
    macro_rules! comb {
        ($i1:literal, $i2:literal, $i3:literal) => {
            [c[$i1 - 1], c[$i2 - 1], c[$i3 - 1]]
        };
    }
    c.make_contiguous().sort_unstable();
    state.b.make_contiguous();
    if (state.b == comb![1, 2, 3]/* abc */) {
        return;
    }
    if (state.b == comb![1, 3, 2]/* acb */) {
        rrb(state);
        //swap
        rb(state);
    }
    if (state.b == comb![2, 3, 1]/* bca */) {
        rb(state);
        //SWAP
        rb(state);
    }
    if (state.b == comb![2, 1, 3]/* bac */) {
        rb(state);
        //SWAP
        rb(state);
    }
    if (state.b == comb![3, 2, 1]/* cba */) {
        //SWAP
        rb(state);
    }
    if (state.b == comb![3, 1, 2]/* cab */) {
        rrb(state);
    }
}

fn run_with_items(items: impl Iterator<Item = i32>) -> Result<(usize, usize), ()> {
    // ITERATOR is always at most two element !
    let mut state = State {
        a: items.collect(),
        b: VecDeque::with_capacity(1024),
        sorted: VecDeque::new(),
        counts: 0,
        moves: Vec::new(),
    };

    let state_cpy = state.clone();

    // Create the output elements in good order !
    state.sorted = state.a.clone().into_iter().map(|e| (e, false)).collect();
    state
        .sorted
        .make_contiguous()
        .sort_unstable_by_key(|(e, _)| *e);
    state.sorted.make_contiguous();

    // init
    pb(&mut state);
    pb(&mut state);
    pb(&mut state);
    if let Some(e) = state.sorted.iter_mut().find(|(e, _)| *e == state.b[0]) {
        e.1 = true;
    }
    if let Some(e) = state.sorted.iter_mut().find(|(e, _)| *e == state.b[1]) {
        e.1 = true;
    }
    if let Some(e) = state.sorted.iter_mut().find(|(e, _)| *e == state.b[2]) {
        e.1 = true;
    }
    sort_three_b(&mut state);
    // if state.b[0] < state.b[1] {
    //     rb(&mut state);
    // }
    // end of init

    // sorting
    while state.a.len() > 1 {
        let best_move = state
            .a
            .iter()
            .enumerate()
            .map(|(index, &elem)| {
                (index, {
                    let mut out = 1;
                    let target_index = (find_place(state.a[index], &state)
                        + (state.b.len() - state.b.iter().position_max().unwrap_or(0)))
                        % state.b.len();
                    let mut rotate_a = target(0, index, state.a.len());
                    let mut rotate_b = target(target_index, 0, state.b.len());

                    if (std::mem::discriminant(&rotate_a) != std::mem::discriminant(&rotate_b)) {
                        let diff_flip_a = {
                            let (Rotates::Forward(a) | Rotates::Reverse(a)) =
                                rotate_a.flip(state.a.len());
                            let (Rotates::Forward(b) | Rotates::Reverse(b)) = rotate_b; //.flip(state.a.len());
                            a.abs_diff(b)
                        };
                        let diff_flip_b = {
                            let (Rotates::Forward(a) | Rotates::Reverse(a)) = rotate_a; // .flip(state.a.len());
                            let (Rotates::Forward(b) | Rotates::Reverse(b)) =
                                rotate_b.flip(state.b.len());
                            a.abs_diff(b)
                        };
                        if (diff_flip_a > diff_flip_b) {
                            rotate_b = rotate_b.flip(state.b.len());
                        } else {
                            rotate_a = rotate_a.flip(state.a.len());
                        }
                    }

                    let (Rotates::Forward(a) | Rotates::Reverse(a)) = rotate_a; // .flip(state.a.len());
                    let (Rotates::Forward(b) | Rotates::Reverse(b)) = rotate_b;
                    for _ in 0..a.min(b) {
                        out += 1;
                    }
                    if a < b {
                        for _ in 0..(b - a) {
                            out += 1;
                        }
                    } else {
                        for _ in 0..(a - b) {
                            out += 1;
                        }
                    }
                    out
                })
            })
            .min_by_key(|(_, i)| *i);
        do_move(&mut state, best_move.map(|t| t.0).unwrap_or_default());
    }

    //let index = find_place(state.a[0], &state);

    // end of sorting
    target(
        0,
        state.b.iter().position_max().unwrap_or_default(),
        state.b.len(),
    )
    .action(&mut state, StackSelector::B);
    let back = *state.a.back().unwrap();
    let mut s = true;
    while !state.b.is_empty() {
        if s && &back > state.b.front().unwrap() {
            ra(&mut state);
            s = false;
        }
        pa(&mut state);
    }

    target(
        0,
        state.a.iter().position_min().unwrap_or_default(),
        state.a.len(),
    )
    .action(&mut state, StackSelector::A);
    // everything should be in the correct place !
    if is_sorted(state.a.iter()) {
        Ok((state.counts, optimize_moves(&mut state, state_cpy)))
    } else {
        println!("{back}");
        dbg!(&state.a);
        Err(())
    }
}
