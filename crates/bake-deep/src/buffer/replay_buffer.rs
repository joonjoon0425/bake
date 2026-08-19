use burn::{Tensor, tensor::Device};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use crate::types::{Batch, Batchable, Transition};

/// Replay Buffer Implementation
pub struct ReplayBuffer<Obs: Batchable, Action: Batchable, Barrier: Batchable, Extra: Batchable = ()> {
    capacity: usize,
    head: usize,

    obss: Vec<Obs>,
    actions: Vec<Action>,
    rewards: Vec<f32>,
    next_obss: Vec<Obs>,

    terminated: Vec<f32>,
    truncated: Vec<f32>,

    barriers: Option<Vec<Barrier>>,
    next_barriers: Option<Vec<Barrier>>,

    extras: Vec<Extra>,

    rng: StdRng,
    device: Device,
}

impl<Obs: Batchable, Action: Batchable, Barrier: Batchable, Extra: Batchable> ReplayBuffer<Obs, Action, Barrier, Extra> {
    pub fn new(seed: u64, capacity: usize, device: Device) -> Self {
        let obss = Vec::with_capacity(capacity);
        let actions = Vec::with_capacity(capacity);
        let rewards = Vec::with_capacity(capacity);
        let next_obss = Vec::with_capacity(capacity);
        let terminated = Vec::with_capacity(capacity);
        let truncated = Vec::with_capacity(capacity);
        let barriers = None;
        let next_barriers = None;
        let extras = Vec::with_capacity(capacity);
        
        Self {
            obss,
            actions,
            rewards,
            next_obss,
            terminated,
            truncated,
            barriers,
            next_barriers,

            extras,
            capacity,
            head: 0,
            rng: StdRng::seed_from_u64(seed),
            device,
        }
    }

    pub fn push(&mut self, t: Transition<Obs, Action, Barrier, Extra>) {
        if self.len() < self.capacity {
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
                    let mut barriers = Vec::with_capacity(self.capacity);
                    barriers.push(barrier);
                    let mut next_barriers= Vec::with_capacity(self.capacity);
                    next_barriers.push(next_barrier);
                    self.barriers = Some(barriers);
                    self.next_barriers = Some(next_barriers);
                },
                _ => { panic!("The environment requires to 1. always give a mask or 2. always do not give a mask") }
            }
            self.extras.push(t.extra);
        } else {
            self.obss[self.head] = t.obs;
            self.actions[self.head] = t.action;
            self.rewards[self.head] = t.reward;
            self.next_obss[self.head] = t.next_obs;
            self.terminated[self.head] = if t.terminated { 1f32 } else { 0f32 };
            self.truncated[self.head] = if t.truncated { 1f32 } else { 0f32 };
            match (&mut self.barriers, &mut self.next_barriers, t.barrier, t.next_barrier) {
                (Some(barriers), Some(next_barriers), Some(barrier), Some(next_barrier)) => {
                    barriers[self.head] = barrier;
                    next_barriers[self.head] = next_barrier;
                },
                (None, None, None, None) => {},
                _ => { panic!("The environment requires to 1. always give a mask or 2. always do not give a mask") }
            }
            self.extras[self.head] = t.extra;
        }

        self.head = (self.head + 1) % self.capacity;
    }

    pub fn len(&self) -> usize { self.obss.len() }

    pub fn sample(&mut self, batch_size: usize) -> Option<Batch<Obs, Action, Barrier, Extra>> {
        let len = self.len();

        if len < batch_size { return None; }

        let indices: Vec<usize> = (0..batch_size).map(|_| self.rng.random_range(0..len)).collect();

        let (o, a, r, no, te, tr, ex): (Vec<Obs>, Vec<Action>, Vec<f32>, Vec<Obs>, Vec<f32>, Vec<f32>, Vec<Extra>)
            = indices.iter().map(|&index| {(
                    self.obss[index].clone(),
                    self.actions[index].clone(),
                    self.rewards[index],
                    self.next_obss[index].clone(),
                    self.terminated[index],
                    self.truncated[index],
                    self.extras[index].clone())
            }).collect();
        
        let (barriers, next_barriers) = match (&mut self.barriers, &mut self.next_barriers) {
            (Some(b), Some(nb)) => {
                let (barriers, next_barriers) = indices.iter().map(|&index| {
                    (
                        b[index].clone(),
                        nb[index].clone(),
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

        Some(Batch {
            obss: Obs::batch(o, &self.device),
            actions: Action::batch(a, &self.device),
            rewards: Tensor::from_floats(r.as_slice(), &self.device),
            next_obss: Obs::batch(no, &self.device),
            terminated: Tensor::from_floats(te.as_slice(), &self.device),
            truncated: Tensor::from_floats(tr.as_slice(), &self.device),
            barriers,
            next_barriers,
            extras: Extra::batch(ex, &self.device),

            batch_size,
        })
    }
}