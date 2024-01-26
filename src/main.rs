#![allow(dead_code)]
#![allow(unused)]
use std::collections::{vec_deque::VecDeque, HashSet};

use itertools::Itertools;
use rand::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use statrs::statistics::{Distribution, *};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
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

struct State {
    a: Stack,
    b: Stack,
    sorted: Stack<(i32, bool)>,
    counts: usize,
    output: Box<dyn std::io::Write>,
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

impl std::ops::Neg for Rotates {
    type Output = Self;
    fn neg(self) -> Self::Output {
        match self {
            Self::Reverse(n) => Self::Forward(n),
            Self::Forward(n) => Self::Reverse(n),
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
            Self::Reverse(_) => match stack {
                StackSelector::A => rra,
                StackSelector::B => rrb,
            },
            Self::Forward(_) => match stack {
                StackSelector::A => ra,
                StackSelector::B => rb,
            },
        };
        let (Self::Forward(n) | Self::Reverse(n)) = *self;
        for _ in 0..n {
            (func)(state);
        }
    }
}

fn do_move(state: &mut State, index: usize) {
    let target_index = find_place(state.a[index], state);
    let rotate_a = target(0, index, state.a.len());
    let rotate_b = target(target_index, 0, state.b.len());
    /*
    println!("============================================================================");
    dbg!(&state.b);
    println!(
        "rotating A by {rotate_a:?} to get number {} at index 0 from index {index}",
        state.a[index]
    );
    println!(
        "rotating B by {rotate_b:?} to get number {} at index {target_index} from index 0",
        state.a[index]
    );
    */
    /*match rotate_a {
        Rotates::Forward(n) => state.a.rotate_right(n),
        Rotates::Reverse(n) => state.a.rotate_left(n),
    }
    match rotate_b {
        Rotates::Forward(n) => state.b.rotate_right(n),
        Rotates::Reverse(n) => state.b.rotate_left(n),
    }*/
    //dbg!(&state.a);
    rotate_a.action(state, StackSelector::A);
    rotate_b.action(state, StackSelector::B);
    pb(state);
    //state.b.push_front(state.a.pop_front().unwrap());
    //dbg!(&state.b);
    if let Some(e) = state.sorted.iter_mut().find(|(e, _)| *e == state.b[0]) {
        e.1 = true;
    }
    target(
        0,
        state.b.iter().position_min().unwrap_or_default(),
        state.b.len(),
    )
    .action(state, StackSelector::B)
}

const PRINT_ACTIONS: bool = false;

macro_rules! print_action {
    ($state:expr, $($t:tt)*) => {
        if PRINT_ACTIONS {
            let writer: &mut dyn std::io::Write = &mut $state.output;
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
        .max(1024);

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
                .into_iter()
                .map(Result::unwrap)
                .map(|i| i as f64)
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

fn run_with_items(items: impl Iterator<Item = i32>) -> Result<usize, ()> {
    // ITERATOR is always at most two element !
    let mut state = State {
        a: items.collect(),
        b: VecDeque::with_capacity(1024),
        sorted: VecDeque::new(),
        counts: 0,
        output: Box::new(std::io::sink()),
        moves: Vec::new(),
    };

    // Create the output elements in good order !
    state.sorted = state.a.clone().into_iter().map(|e| (e, false)).collect();
    state
        .sorted
        .make_contiguous()
        .sort_unstable_by_key(|(e, _)| *e);
    state.sorted.make_contiguous().reverse();

    // init
    pb(&mut state);
    pb(&mut state);
    if let Some(e) = state.sorted.iter_mut().find(|(e, _)| *e == state.b[0]) {
        e.1 = true;
    }
    if let Some(e) = state.sorted.iter_mut().find(|(e, _)| *e == state.b[1]) {
        e.1 = true;
    }
    if state.b[0] > state.b[1] {
        rb(&mut state);
    }
    // end of init

    // sorting
    while !state.a.is_empty() {
        let best_move = state
            .a
            .iter()
            .enumerate()
            .map(|(index, &elem)| {
                (index, {
                    let swap_a = target(index, 0, state.a.len());
                    let swap_b = target(find_place(elem, &state), 0, state.b.len());
                    swap_a.max(swap_b)
                })
            })
            .min_by_key(|(_, i)| *i);
        do_move(&mut state, best_move.map(|t| t.0).unwrap_or_default());
    }

    // end of sorting
    target(
        0,
        state.b.iter().position_min().unwrap_or_default(),
        state.b.len(),
    )
    .action(&mut state, StackSelector::B);

    while !state.b.is_empty() {
        pa(&mut state);
        rra(&mut state);
    }

    // everything should be in the correct place !
    if is_sorted(state.a.iter()) {
        Ok(state.counts)
    } else {
        Err(())
    }
}
