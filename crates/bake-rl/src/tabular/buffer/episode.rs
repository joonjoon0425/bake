//! implementation of episode buffer for Monte Carlo methods
use crate::core::data::Transition;

/// Episode buffer
/// - collects trajectories
pub struct EpisodeBuffer<Obs: Clone, Constraint: Clone, Action: Clone, Reward: Clone> {
    episode: Vec<Transition<Obs, Constraint, Action, Reward>>,
    finished: bool,
}

impl<Obs: Clone, Constraint: Clone, Action: Clone, Reward: Clone> EpisodeBuffer<Obs, Constraint, Action, Reward> {
    /// create a new empty `EpisodeBuffer`
    pub fn new() -> Self {
        Self {
            episode: vec![],
            finished: false
        }
    }

    /// return the current length of buffer
    pub fn len(&self) -> usize {
        self.episode.len()
    }

    /// push one transition into buffer
    pub fn push(&mut self, t: Transition<Obs, Constraint, Action, Reward>) {
        self.finished = t.terminated || t.truncated;
        self.episode.push(t);
    }

    /// pop all elements of buffer as batch
    pub fn pop(&mut self) -> Vec<Transition<Obs, Constraint, Action, Reward>> {
        let vec = std::mem::replace(&mut self.episode, vec![]);
        self.finished = false;
        vec
    }

    /// checks if buffer is ready
    /// - `true` if it has collected any finished episode (truncated or terminated)
    pub fn is_ready(&self) -> bool {
        self.finished
    }
}