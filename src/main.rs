#![allow(dead_code)]
#![allow(unused)]
use std::{
    collections::{vec_deque::VecDeque, HashSet},
    marker::PhantomData,
    ops::Add,
};

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
    SwapA,
    SwapB,
    SwapBoth,
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
            Move::RotateBoth => rr(&mut cpy),
            Move::RevRotateA => rra(&mut cpy),
            Move::RevRotateB => rrb(&mut cpy),
            Move::RevRotateBoth => rrr(&mut cpy),
            Move::SwapA => sa(&mut cpy),
            Move::SwapB => sb(&mut cpy),
            Move::SwapBoth => ss(&mut cpy),
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
        let first = span.first().unwrap();
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

fn is_sorted<I>(data: I) -> bool
where
    I: IntoIterator,
    I::Item: Ord + Clone,
{
    data.into_iter().tuple_windows().all(|(a, b)| a <= b)
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
        let target = match iter_size {
            500 => 5500f64,
            100 => 700f64,
            _ => return,
        };
        let over_target = data.iter().filter(|d| **d >= target).count();
        println!(
            "{} ({:02.1}) runs are over the target of {}",
            over_target,
            over_target as f64 / iter_numbers as f64 * 100.0,
            target as usize
        );

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

const PRINT_ACTIONS: bool = false;

macro_rules! print_action {
    ($state:expr, $($t:tt)*) => {
        if PRINT_ACTIONS {
            let writer: &mut dyn std::io::Write = &mut std::io::stdout();
            write!(writer, $($t)*).unwrap();
        }
    };
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

fn sa(state: &mut State) {
    let sec = state.a.pop_front();
    let fir = state.a.pop_front();
    if let Some(e) = sec {
        state.a.push_front(e);
    }
    if let Some(e) = fir {
        state.a.push_front(e);
    }
    print_action!(state, "SA");
    state.counts += 1;
    state.moves.push(Move::SwapA)
}

fn sb(state: &mut State) {
    let sec = state.b.pop_front();
    let fir = state.b.pop_front();
    if let Some(e) = sec {
        state.b.push_front(e);
    }
    if let Some(e) = fir {
        state.b.push_front(e);
    }
    print_action!(state, "SB");
    state.counts += 1;
    state.moves.push(Move::SwapB)
}
fn ss(state: &mut State) {
    let sec = state.b.pop_front();
    let fir = state.b.pop_front();
    if let Some(e) = sec {
        state.b.push_front(e);
    }
    if let Some(e) = fir {
        state.b.push_front(e);
    }
    let sec = state.a.pop_front();
    let fir = state.a.pop_front();
    if let Some(e) = sec {
        state.a.push_front(e);
    }
    if let Some(e) = fir {
        state.a.push_front(e);
    }
    print_action!(state, "SS");
    state.counts += 1;
    state.moves.push(Move::SwapBoth)
}

fn sort_three(state: &mut State, selector: StackSelector, min_first: bool) {
    macro_rules! stack {
        () => {
            match selector {
                StackSelector::B => &mut state.b,
                StackSelector::A => &mut state.a,
            }
        };
    }
    let [swap, rotate, rev_rotate] = match selector {
        StackSelector::A => [sa, ra, rra],
        StackSelector::B => [sb, rb, rrb],
    };
    match stack!().len() {
        0 | 1 | 4.. => return,
        2 => {
            let func = match min_first {
                true => PartialOrd::gt,
                false => PartialOrd::lt,
            };

            if func(&stack!()[0].clone(), &stack!()[1]) {
                swap(state);
            }
            return;
        }
        3 => {}
    }
    let mut c = stack!().clone();
    macro_rules! comb {
        ($i1:literal, $i2:literal, $i3:literal) => {
            &[c[$i1 - 1], c[$i2 - 1], c[$i3 - 1]]
        };
    }
    c.make_contiguous().sort_unstable();
    if min_first {
        c.make_contiguous().reverse();
    }
    stack!().make_contiguous();

    if (stack!() == comb![1, 2, 3]/* abc */) {
        swap(state);
        rotate(state);
    } else if (stack!() == comb![1, 3, 2]/* acb */) {
        rev_rotate(state);
    } else if (stack!() == comb![2, 3, 1]/* bca */) {
        swap(state);
    } else if (stack!() == comb![2, 1, 3]/* bac */) {
        rev_rotate(state);
    } else if (stack!() == comb![3, 2, 1]/* cba */) {
    } else if (stack!() == comb![3, 1, 2]/* cab */) {
        rev_rotate(state);
        swap(state);
        rotate(state);
    }
}

fn fuck_mut<'a, 'b, T>(r: &'a mut T) -> &'b mut T {
    unsafe { std::mem::transmute(r) }
}

fn do_move(state: &mut State, index: usize) {
    run_func_with_best_rotate_for_item(
        index,
        fuck_mut(state),
        StackSelector::A,
        false,
        RotationData {
            args: fuck_mut(state),
            main_forward: |s| ra(*s),
            main_reverse: |s| rra(*s),
            dual_forward: |s| rr(*s),
            dual_reverse: |s| rrr(*s),
            other_forward: |s| rb(*s),
            other_reverse: |s| rrb(*s),
        },
    );

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

fn run_func_with_best_rotate_for_item<
    Args,
    F1: Fn(&mut Args),
    F2: Fn(&mut Args),
    F3: Fn(&mut Args),
    F4: Fn(&mut Args),
    F5: Fn(&mut Args),
    F6: Fn(&mut Args),
>(
    index: usize,
    state: &State,
    pop_from: StackSelector,
    min_zero_pos: bool,
    mut data: RotationData<Args, F1, F2, F3, F4, F5, F6>,
) {
    macro_rules! stack {
        (main) => {
            match pop_from {
                StackSelector::A => &state.a,
                StackSelector::B => &state.b,
            }
        };
        (inv) => {
            match pop_from {
                StackSelector::B => &state.a,
                StackSelector::A => &state.b,
            }
        };
    }
    let find_func = match min_zero_pos {
        false => itertools::Itertools::position_max,
        true => itertools::Itertools::position_min,
    };

    let target_index = (find_place(stack!(main)[index], state)
        + (stack!(inv).len() - find_func(stack!(inv).iter()).unwrap_or(0)))
        % stack!(inv).len();
    let mut rotate_main = target(0, index, stack!(main).len());
    let mut rotate_inv = target(target_index, 0, stack!(inv).len());

    macro_rules! choose_rot {
        (main) => {
            |s| match rotate_main {
                Rotates::Forward(_) => (data.main_forward)(s),
                Rotates::Reverse(_) => (data.main_reverse)(s),
            }
        };
        (both) => {
            |s| match rotate_main {
                Rotates::Forward(_) => (data.dual_forward)(s),
                Rotates::Reverse(_) => (data.dual_reverse)(s),
            }
        };
        (inv) => {
            |s| match rotate_inv {
                Rotates::Forward(_) => (data.other_forward)(s),
                Rotates::Reverse(_) => (data.other_reverse)(s),
            }
        };
    }

    if (std::mem::discriminant(&rotate_main) != std::mem::discriminant(&rotate_inv)) {
        let diff_flip_main = {
            let (Rotates::Forward(main) | Rotates::Reverse(main)) =
                rotate_main.flip(stack!(main).len());
            let (Rotates::Forward(inv) | Rotates::Reverse(inv)) = rotate_inv;
            main.abs_diff(inv)
        };
        let diff_flip_inv = {
            let (Rotates::Forward(main) | Rotates::Reverse(main)) = rotate_main;
            let (Rotates::Forward(inv) | Rotates::Reverse(inv)) =
                rotate_inv.flip(stack!(inv).len());
            main.abs_diff(inv)
        };
        let diff_no_flip = {
            let (Rotates::Forward(main) | Rotates::Reverse(main)) = rotate_main;
            let (Rotates::Forward(inv) | Rotates::Reverse(inv)) = rotate_inv;
            main + inv
        };
        match diff_no_flip.min(diff_flip_inv).min(diff_flip_main) {
            n if n == diff_no_flip => {}
            n if n == diff_flip_main => {
                rotate_main = rotate_main.flip(stack!(main).len());
            }
            n if n == diff_flip_inv => {
                rotate_inv = rotate_inv.flip(stack!(inv).len());
            }
            _ => {}
        }
    }
    if std::mem::discriminant(&rotate_main) == std::mem::discriminant(&rotate_inv) {
        let (Rotates::Forward(main) | Rotates::Reverse(main)) = rotate_main;
        let (Rotates::Forward(inv) | Rotates::Reverse(inv)) = rotate_inv;

        for _ in 0..(main.min(inv)) {
            choose_rot!(both)(&mut data.args);
        }
        if main > inv {
            for _ in 0..(main - inv) {
                choose_rot!(main)(&mut data.args);
            }
        } else {
            for _ in 0..(inv - main) {
                choose_rot!(inv)(&mut data.args);
            }
        }
    } else {
        let (Rotates::Forward(main) | Rotates::Reverse(main)) = rotate_main;
        let (Rotates::Forward(inv) | Rotates::Reverse(inv)) = rotate_inv;
        for _ in 0..main {
            choose_rot!(main)(&mut data.args);
        }
        for _ in 0..inv {
            choose_rot!(inv)(&mut data.args);
        }
    }
}

struct RotationData<
    Args,
    F1: Fn(&mut Args),
    F2: Fn(&mut Args),
    F3: Fn(&mut Args),
    F4: Fn(&mut Args),
    F5: Fn(&mut Args),
    F6: Fn(&mut Args),
> {
    args: Args,
    dual_forward: F1,
    dual_reverse: F2,
    main_forward: F3,
    main_reverse: F4,
    other_forward: F5,
    other_reverse: F6,
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
    for item in &state.b {
        if let Some(e) = state.sorted.iter_mut().find(|(e, _)| e == item) {
            e.1 = true;
        }
    }
    sort_three(&mut state, StackSelector::B, false);
    //dbg!(&state.b);
    // if state.b[0] < state.b[1] {
    //     rb(&mut state);
    // }
    // end of init
    // sorting
    //println!("before_sorting");
    while state.a.len() > 3 {
        let best_move = (0..(state.a.len()))
            .map(|index| {
                let mut out = 0;
                let func = |i: &mut &mut i32| {
                    **i += 1;
                };
                run_func_with_best_rotate_for_item(
                    index,
                    &state,
                    StackSelector::A,
                    false,
                    RotationData {
                        args: &mut out,
                        main_forward: func,
                        main_reverse: func,
                        dual_forward: func,
                        dual_reverse: func,
                        other_forward: func,
                        other_reverse: func,
                    },
                );
                out
            })
            .position_min();
        do_move(&mut state, best_move.unwrap_or(0));
    }

    //println!("before merging: {}", state.counts);
    sort_three(&mut state, StackSelector::A, true);
    //let index = find_place(state.a[0], &state);

    // end of sorting
    target(
        0,
        state.b.iter().position_max().unwrap_or_default(),
        state.b.len(),
    )
    .action(&mut state, StackSelector::B);

    state.sorted.make_contiguous().reverse();
    state.sorted.iter_mut().for_each(|t| t.1 = false);

    for item in &state.a {
        if let Some(e) = state.sorted.iter_mut().find(|(e, _)| e == item) {
            e.1 = true;
        }
    }
    //dbg!(&state.a);
    //dbg!(&state.b);
    while !state.b.is_empty() {
        //println!("=================");
        //target(
        //    0,
        //    state.a.iter().position_min().unwrap_or_default(),
        //    state.a.len(),
        //)
        //.action(&mut state, StackSelector::A);
        //println!("yaaas");
        let idx = (0..(state.b.len()))
            .map(|index| {
                let mut out = 0;
                let func = |i: &mut &mut i32| {
                    **i += 1;
                };
                run_func_with_best_rotate_for_item(
                    index,
                    &state,
                    StackSelector::B,
                    true,
                    RotationData {
                        args: &mut out,
                        main_forward: func,
                        main_reverse: func,
                        dual_forward: func,
                        dual_reverse: func,
                        other_forward: func,
                        other_reverse: func,
                    },
                );
                out
                //find_best_rotate_for_item(index, &state, StackSelector::B, true);
            })
            .position_min()
            .unwrap_or(0);
        run_func_with_best_rotate_for_item(
            idx,
            fuck_mut(&mut state),
            StackSelector::B,
            true,
            RotationData {
                args: fuck_mut(&mut state),
                main_forward: |s| rb(s),
                main_reverse: |s| rrb(s),
                dual_forward: |s| rr(s),
                dual_reverse: |s| rrr(s),
                other_forward: |s| ra(s),
                other_reverse: |s| rra(s),
            },
        );
        pa(&mut state);
        for item in &state.a {
            if let Some(e) = state.sorted.iter_mut().find(|(e, _)| e == item) {
                e.1 = true;
            }
        }
    }
    //println!("after_sorting");

    /*let back = *state.a.back().unwrap();
        let mut s = true;
        while !state.b.is_empty() {
            if s && &back > state.b.front().unwrap() {
                ra(&mut state);
                s = false;
            }
            pa(&mut state);
        }
    */
    let mut rotation = target(
        0,
        state.a.iter().position_min().unwrap_or_default(),
        state.a.len(),
    );

    if true {
        let (Rotates::Forward(normal) | Rotates::Reverse(normal)) = rotation;
        let (Rotates::Forward(flipped) | Rotates::Reverse(flipped)) = rotation.flip(state.a.len());
        if (normal > flipped) {
            rotation = rotation.flip(state.a.len());
        }
    }

    rotation.action(&mut state, StackSelector::A);
    // everything should be in the correct place !
    if is_sorted(state.a.iter()) {
        Ok((state.counts, optimize_moves(&mut state, state_cpy)))
    } else {
        //println!("{back}");
        dbg!(&state.a);
        Err(())
    }
}
