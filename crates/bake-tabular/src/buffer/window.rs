//! Window buffer for n-step methods
//! 

use std::collections::VecDeque;

use crate::{constraint::Constraint, data::Transition};

/// Window buffer for n step methods
pub struct WindowBuffer<C: Constraint> {
    data: VecDeque<Transition<C>>,
}

impl<C: Constraint> WindowBuffer<C> {
    /// create a new empty window buffer
    pub fn new() -> Self { Self { data: VecDeque::new() } }

    /// return the current length of the buffer
    pub fn len(&self) -> usize { self.data.len() } 

    /// push a new data into buffer
    pub fn push(&mut self, t: Transition<C>) { self.data.push_back(t); }

    /// provides a vector of all transitions in current buffer
    /// # Warning
    /// - `sample` will delete the first element after being called 
    pub fn sample(&mut self) -> Vec<Transition<C>> {
        let mut vec = Vec::with_capacity(self.len());
        for t in &self.data {
            vec.push(t.clone());
        }
        vec
    }
    /// clear the buffer
    pub fn clear(&mut self) { self.data.clear(); }
}