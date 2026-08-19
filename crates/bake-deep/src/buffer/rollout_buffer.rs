use burn::{Tensor, tensor::Device};

use crate::types::{Batch, Batchable, Transition};

/// Rollout Buffer Implementation. Also works as Episode Buffer if n = None is given
pub struct RolloutBuffer<Obs: Batchable, Action: Batchable, Barrier: Batchable, Extra: Batchable = ()>{
    obss: Vec<Obs>,
    actions: Vec<Action>,
    rewards: Vec<f32>,
    next_obss: Vec<Obs>,
    terminated: Vec<f32>,
    truncated: Vec<f32>,

    barriers: Option<Vec<Barrier>>,
    next_barriers: Option<Vec<Barrier>>,
    extras: Vec<Extra>,

    device: Device,

    n: Option<usize>,
}

impl<Obs: Batchable, Action: Batchable, Barrier: Batchable, Extra: Batchable> RolloutBuffer<Obs, Action, Barrier, Extra> {
    pub fn new(n: Option<usize>, device: Device) -> Self {
        Self {
            obss: vec![],
            actions: vec![],
            rewards: vec![],
            next_obss: vec![],
            terminated: vec![],
            truncated: vec![],
            barriers: None,
            next_barriers: None,
            extras: vec![],

            device,
            n,
        }
    }

    pub fn len(&self) -> usize {
        self.obss.len()
    }

    pub fn is_full(&self) -> bool {
        let len = self.len();
        if let Some(n) = self.n {
            return len == n;
        } else {
            return self.terminated[len - 1] == 1f32 || self.truncated[len - 1] == 1f32;
        }
    }

    pub fn push(&mut self, t: Transition<Obs, Action, Barrier, Extra>) {
        self.obss.push(t.obs);
        self.actions.push(t.action);
        self.rewards.push(t.reward);
        self.next_obss.push(t.next_obs);
        self.terminated.push(if t.terminated { 1f32 } else { 0f32 });
        self.truncated.push(if t.truncated { 1f32 } else { 0f32 });
        let len = self.len();
            match (&mut self.barriers, &mut self.next_barriers, t.barrier, t.next_barrier, len) {
                (Some(barriers), Some(next_barriers), Some(barrier), Some(next_barrier), _) => {
                    barriers.push(barrier);
                    next_barriers.push(next_barrier);
                },
                (None, None, None, None, _) => {},
                (None, None, Some(barrier), Some(next_barrier), 1) => {
                    let mut barriers = vec![];
                    barriers.push(barrier);
                    let mut next_barriers= vec![];
                    next_barriers.push(next_barrier);
                    self.barriers = Some(barriers);
                    self.next_barriers = Some(next_barriers);
                },
                _ => { panic!("The environment requires to 1. always give a mask or 2. always do not give a mask") }
            }
        self.extras.push(t.extra);
    }

    pub fn pop(&mut self) -> Batch<Obs, Action, Barrier, Extra> {
        let (barriers, next_barriers) = match (&mut self.barriers, &mut self.next_barriers) {
            (Some(b), Some(nb)) => {
                let (barriers, next_barriers) = b.iter().zip(nb.iter()).map(|(b, nb)| {
                    (
                        b.clone(),
                        nb.clone(),
                    )
                }).collect();
                let barriers = Barrier::batch(barriers, &self.device);
                let next_barriers = Barrier::batch(next_barriers, &self.device);
                (Some(barriers), Some(next_barriers))
            },
            (None, None) => {
                (None, None)
            },
            _ => { panic!("Barrier and next barrier value does not coincides") }
        };

        let batched_steps = Batch {
            obss: Obs::batch(self.obss.clone(), &self.device),
            actions: Action::batch(self.actions.clone(), &self.device),
            rewards: Tensor::from_floats(self.rewards.as_slice(), &self.device),
            next_obss: Obs::batch(self.next_obss.clone(), &self.device),
            terminated: Tensor::from_floats(self.terminated.as_slice(), &self.device),
            truncated: Tensor::from_floats(self.truncated.as_slice(), &self.device),
            barriers,
            next_barriers,
            extras: Extra::batch(self.extras.clone(), &self.device),

            batch_size: self.len()
        };

        self.obss.clear();
        self.actions.clear();
        self.rewards.clear();
        self.next_obss.clear();
        self.terminated.clear();
        self.truncated.clear();

        match (&mut self.barriers, &mut self.next_barriers) {
            (Some(b), Some(nb)) => { b.clear(); nb.clear(); },
            (None, None) => {},
            _ => { panic!("Barrier and next barrier value does not coincides") }
        };

        self.extras.clear();

        batched_steps
    }
}