#![allow(dead_code)]
use std::collections::vec_deque::VecDeque;

use itertools::Itertools;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
enum Move {
    SwapA,
    SwapB,
    SwapBoth,
    RevSwapA,
    RevSwapB,
    RevSwapBoth,
}

type Stack = VecDeque<i32>;

#[derive(Debug, Clone)]
struct State {
    a: Stack,
    b: Stack,
}

fn target(from: usize, to: usize, stack_len: usize) -> isize {
    let m = (to as isize - from as isize + stack_len as isize) % stack_len as isize;
    //eprintln!("{from} -> {to} = {m}");
    m
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Copy)]
pub enum Rotates {
    Forward(usize),
    Reverse(usize),
}

fn find_place(elem: i32, stack: &mut Stack) -> usize {
    dbg!(elem);
    let (max, min) = (*stack.iter().max().unwrap(), *stack.iter().min().unwrap());
    if elem > max {
        return 0;
    };
    if elem < min {
        return stack.len();
    };
    stack
        .make_contiguous()
        .windows(2)
        .enumerate()
        .inspect(|b| {})
        .find(|(_, w)| w[0] > elem && elem > w[1])
        .map(|(i, _)| i + 1)
        .unwrap()
}

fn do_move(state: &mut State, index: usize) {
    let target_index = find_place(state.a[index], &mut state.b);

    let rotate_a = if state.a.len() / 2 > index {
        Rotates::Forward(index)
    } else {
        Rotates::Reverse(index - state.a.len() / 2)
    };

    let rotate_b = if state.b.len() / 2 > target_index {
        Rotates::Forward(target_index)
    } else {
        Rotates::Reverse(target_index - state.b.len() / 2)
    };

    dbg!(rotate_a);
    dbg!(rotate_b);
    match rotate_a {
        Rotates::Forward(n) => state.a.rotate_right(n),
        Rotates::Reverse(n) => state.a.rotate_left(n),
    }
    match rotate_b {
        Rotates::Forward(n) => state.b.rotate_right(n),
        Rotates::Reverse(n) => state.b.rotate_left(n),
    }
    state.b.push_back(state.a.pop_front().unwrap());
}

fn main() {
    let mut state = State {
        a: VecDeque::with_capacity(1024),
        b: VecDeque::with_capacity(1024),
    };

    if std::env::args()
        .skip(1)
        .map(|s| s.parse::<i32>())
        .try_for_each(|e| {
            e.map(|i| {
                state.a.push_back(i);
            })
        })
        .is_err()
    {
        eprintln!("Error:\nInvalid arguments !");
        std::process::exit(1);
    }
    if !state.a.iter().all_unique() {
        eprintln!("Error:\nDuplicate numbers !");
        std::process::exit(1);
    }
    dbg!(&state.a);
    if state.a.len() >= 2 {
        state.b.push_front(state.a.pop_front().unwrap());
        state.b.push_front(state.a.pop_front().unwrap());
    }
    while state.a.len() > 2 {
        let (max, min) = (
            *state.b.iter().max().unwrap(),
            *state.b.iter().min().unwrap(),
        );
        let best_move = state
            .a
            .iter()
            .enumerate()
            .map(|(index, &elem)| {
                (index, {
                    if elem > max || elem < min {
                        target(index, 0, state.a.len()).abs()
                    } else {
                        let swap_a = target(index, 0, state.a.len());
                        let swap_b = target(0, find_place(elem, &mut state.b), state.b.len());
                        swap_a.abs().max(swap_b.abs())
                    }
                })
            })
            .min_by_key(|(_, i)| i.abs());
        do_move(&mut state, best_move.unwrap().0);
        dbg!(&state.a);
        dbg!(&state.b);
    }
}
