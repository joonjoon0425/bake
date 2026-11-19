//! Window buffer for n-step methods
use std::collections::VecDeque;
use crate::core::data::Transition;

/// window buffer for n-step methods
/// - Saves `n` transitions, and when the new element is pushed, the head element is dropped and the new element locates at tail
pub struct WindowBuffer<Obs: Clone, Constraint: Clone, Action: Clone, Reward: Clone> {
    transitions: VecDeque<Transition<Obs, Constraint, Action, Reward>>,
    n: usize,
}

impl<Obs: Clone, Constraint: Clone, Action: Clone, Reward: Clone> WindowBuffer<Obs, Constraint, Action, Reward> {
    /// create new `WindowBuffer` which saves `n` elements
    pub fn new(n: usize) -> Self {
        Self {
            transitions: VecDeque::with_capacity(n),
            n
        }
    }
    
    /// get length of buffer
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// check if the buffer is ready
    /// - `true` if `self.len() == self.n`
    /// - else `false`
    pub fn is_ready(&self) -> bool {
        self.transitions.len() == self.n
    }

    /// push en element
    /// - If the length of buffer is `n`, first element is dropped and given element is pushed at the end
    /// - Else, given element is pushed at the end
    pub fn push(&mut self, transition: Transition<Obs, Constraint, Action, Reward>) {
        if self.len() == self.n {
            self.transitions.pop_front();
        }
        self.transitions.push_back(transition);
    }

    /// peek the elements
    /// # Note
    /// - `clone` the whole stored element an give it as `Vec`
    pub fn peek(&self) -> Vec<Transition<Obs, Constraint, Action, Reward>> {
        self.transitions.iter().cloned().collect()
    }

    /// pop the front element
    pub fn pop(&mut self) -> Option<Transition<Obs, Constraint, Action, Reward>> { self.transitions.pop_front() }

    /// clear the buffer
    pub fn clear(&mut self) { self.transitions.clear() }

    /// drain the buffer
    /// # Warning
    /// - the elements are given as `impl Iterator`, but in a bit different way
    /// - suppose that the buffer has 4 elements left. then the iterator gives the 4 elements as `Vec` first, and then the last 3 elements as `Vec`, 2 elements as `Vec`, 1 elements as `Vec` and ends
    /// - this is to ease the update using leftover n steps, for n-step algorithms
    pub fn drain(&mut self) -> impl Iterator<Item = Vec<Transition<Obs, Constraint, Action, Reward>>> {
        let iter: Vec<_> = (0..self.transitions.len()).map(|_| {
            let vec: Vec<_> = self.transitions.iter().cloned().collect();
            self.transitions.pop_front();
            vec
        }).collect();
        self.transitions.clear();
        iter.into_iter()
    } 
}