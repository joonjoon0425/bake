//! Buffer for n-step methods
//! 
use std::collections::VecDeque;
use crate::{constraint::Constraint, data::Transition};

/// Buffer for atomic n-step methods
pub struct AtomicNStepBuffer<C: Constraint> {
    window: VecDeque<Transition<C>>,
    n: usize,
    gamma: f32,
}

impl<C: Constraint> AtomicNStepBuffer<C> {
    /// create a new `n`-step `AtomicNStepBuffer` 
    pub fn new(n: usize, gamma: f32) -> Self {
        Self {
            window: VecDeque::with_capacity(n),
            n,
            gamma
        }
    }
    /// push a new transition to the buffer
    pub fn push(&mut self, t: Transition<C>) {
        self.window.push_back(t);
    }
    /// pop an n-step transition: reward -> n-step reward, next_obs -> obs after n-step
    /// # Warning
    /// - The buffer must contain the termination transition only in the last item
    /// - That is, the user must call `drain` after the termination
    /// - After the function call, the first item of the window buffer is discarded
    pub fn pop(&mut self) -> Transition<C>{
        let off_policy = self.window.front().unwrap().get("prob").is_some();
        let mut reward = 0f32;
        for t in self.window.iter().rev() {
            reward += self.gamma * reward + t.reward;
            
        }

        if off_policy {
            let mut prob_sum = 0f32;
            for t in self.window.iter().skip(1).rev() {
                prob_sum += t.get("prob")
                    .expect("Off policy method must have the behaviour policy's probability in extra data");
            }
            self.window.back_mut().unwrap().insert("prob_sum", prob_sum);
        }
        let n = self.window.len();
        self.window.back_mut().unwrap().insert("n", n as f32);

        let front = self.window.front().unwrap();
        let back = self.window.back().unwrap();
        let t = Transition {
            obs: front.obs,
            action: front.action,
            reward,
            next_obs: back.next_obs,
            constraint: front.constraint,
            next_constraint: back.next_constraint,
            terminated: back.terminated,
            truncated: back.truncated,
            // has to be fixed; maybe I should add the `next` tag for the extra members
            extra: back.extra.clone(),
        };
        self.window.pop_front();
        t
    }
    /// checks if `pop` is possible
    pub fn is_ready(&self) -> bool { if self.window.len() >= self.n { true } else { false } }
    /// gives the leftover n-step transitions as iterator, clearing the buffer
    pub fn drain(&mut self) -> impl Iterator<Item = Transition<C>> + '_ {
        let mut items = vec![];
        while self.window.len() > 0 {
            items.push(self.pop())
        }
        items.into_iter()
    }
    /// clears the buffer
    pub fn clear(&mut self) { self.window.clear() }
}