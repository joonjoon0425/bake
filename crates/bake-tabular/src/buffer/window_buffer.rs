//! Window buffer for n-step methods
use std::{collections::VecDeque};

use crate::types::{Mask, Transition};

/// A window buffer for n-step methods
pub struct WindowBuffer<M: Mask, Extra: Clone> {
    data: VecDeque<Transition<M, Extra>>,
}

impl<M: Mask, Extra: Clone> WindowBuffer<M, Extra> {
    /// create a new `WindowBuffer` with window size `n`.
    /// pre-allocates `VecDeque` with capacity `n`.
    pub fn new() -> Self {
        Self {
            data: VecDeque::new(),
        }
    }

    /// push a transition to buffer
    pub fn push(&mut self, t: Transition<M, Extra>) {
        self.data.push_back(t);
    }

    /// move a window one step ahead by popping a first element.
    pub fn move_window(&mut self) {
        self.data.pop_front();
    }

    /// give transitions from window buffer.
    pub fn window(&mut self) -> &[Transition<M, Extra>] {
        let t = self.data.make_contiguous();
        t
    }

    /// get current length of buffer
    pub fn len(&self) -> usize {
        self.data.len()
    }
}