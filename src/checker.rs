use std::error::Error;
use crate::Move;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Copy)]
struct InvalidMove;

impl Move {
    fn parse_str(s: String) -> Result<Self, Box<dyn Error>> {
        use Move::*;
        match s.as_str() {
            "pa" => Ok(PushA),
            "pb" => Ok(PushB),
            "sa" => Ok(SwapA),
            "sb" => Ok(SwapB),
            "ss" => Ok(SwapBoth),
            "ra" => Ok(RotateA),
            "rb" => Ok(RotateB),
            "rr" => Ok(RotateBoth),
            "rra" => Ok(RevRotateA),
            "rrb" => Ok(RevRotateB),
            "rrr" => Ok(RevRotateBoth),
            _ => Err("Invalid Move!".into()),
        }
    }
}

pub fn checker_main() -> Result<bool, Box<dyn Error>> {
    let args = std::env::args()
        .skip(1)
        .map(|s| s.parse::<i32>())
        .collect::<Result<Vec<i32>, _>>()?;
    let lines = std::io::stdin()
        .lines()
        .map(|s| {
            s.map_err(|s| Box::new(s) as Box<dyn Error>)
                .and_then(Move::parse_str)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(check(args.into_iter(), lines.into_iter()))
}

pub fn check(input: impl Iterator<Item = i32>, moves: impl Iterator<Item = Move>) -> bool {
    let mut a = input.collect::<Vec<_>>();
    let mut b = Vec::with_capacity(a.capacity());

    for m in moves {
        use Move::*;
        match m {
            RotateA => {
                if !a.is_empty() {
                    let e = a.remove(0);
                    a.push(e);
                }
            }
            RotateB => {
                if !b.is_empty() {
                    let e = b.remove(0);
                    b.push(e);
                }
            }
            RotateBoth => {
                if !a.is_empty() {
                    let e = a.remove(0);
                    a.push(e);
                }
                if !b.is_empty() {
                    let e = b.remove(0);
                    b.push(e);
                }
            }
            RevRotateA => {
                if let Some(e) = a.pop() {
                    a.insert(0, e);
                }
            }
            RevRotateB => {
                if let Some(e) = b.pop() {
                    b.insert(0, e);
                }
            }
            RevRotateBoth => {
                if let Some(e) = a.pop() {
                    a.insert(0, e);
                }
                if let Some(e) = b.pop() {
                    b.insert(0, e);
                }
            }
            PushA => {
                if !b.is_empty() {
                    a.insert(0, b.remove(0))
                }
            }
            PushB => {
                if !a.is_empty() {
                    b.insert(0, a.remove(0))
                }
            }
            SwapA => {
                if a.len() > 1 {
                    unsafe {
                        std::ptr::swap(a.as_mut_ptr().add(0), a.as_mut_ptr().add(1));
                    }
                }
            }
            SwapB => {
                if b.len() > 1 {
                    unsafe {
                        std::ptr::swap(b.as_mut_ptr().add(0), b.as_mut_ptr().add(1));
                    }
                }
            }
            SwapBoth => {
                if a.len() > 1 {
                    unsafe {
                        std::ptr::swap(a.as_mut_ptr().add(0), a.as_mut_ptr().add(1));
                    }
                }
                if b.len() > 1 {
                    unsafe {
                        std::ptr::swap(b.as_mut_ptr().add(0), b.as_mut_ptr().add(1));
                    }
                }
            }
        }
    }

    b.is_empty() && is_sorted(a.iter().copied())
}
use itertools::Itertools;

fn is_sorted<I>(data: I) -> bool
where
    I: IntoIterator,
    I::Item: Ord + Clone,
{
    data.into_iter().tuple_windows().all(|(a, b)| a <= b)
}
