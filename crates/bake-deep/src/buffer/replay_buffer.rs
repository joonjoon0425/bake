use burn::{Tensor, tensor::Device};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use crate::types::{Batch, Batchable, Transition};

/// Replay Buffer Implementation
pub struct ReplayBuffer<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable = ()> {
    capacity: usize,
    head: usize,

    obss: Vec<Obs>,
    actions: Vec<Action>,
    rewards: Vec<f32>,
    next_obss: Vec<Obs>,

    terminated: Vec<f32>,
    truncated: Vec<f32>,

    constraints: Vec<Constraint>,
    next_constraints: Vec<Constraint>,

    extras: Vec<Extra>,

    rng: StdRng,
    device: Device,
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable> ReplayBuffer<Obs, Action, Constraint, Extra> {
    pub fn new(seed: u64, capacity: usize, device: Device) -> Self {
        let obss = Vec::with_capacity(capacity);
        let actions = Vec::with_capacity(capacity);
        let rewards = Vec::with_capacity(capacity);
        let next_obss = Vec::with_capacity(capacity);
        let terminated = Vec::with_capacity(capacity);
        let truncated = Vec::with_capacity(capacity);
        let constraints = Vec::with_capacity(capacity);
        let next_constraints = Vec::with_capacity(capacity);
        let extras = Vec::with_capacity(capacity);
        
        Self {
            obss,
            actions,
            rewards,
            next_obss,
            terminated,
            truncated,
            constraints,
            next_constraints,

            extras,
            capacity,
            head: 0,
            rng: StdRng::seed_from_u64(seed),
            device,
        }
    }

    pub fn push(&mut self, t: Transition<Obs, Action, Constraint, Extra>) {
        if self.len() < self.capacity {
            self.obss.push(t.obs);
            self.actions.push(t.action);
            self.rewards.push(t.reward);
            self.next_obss.push(t.next_obs);
            self.terminated.push(if t.terminated { 1f32 } else { 0f32 });
            self.truncated.push(if t.truncated { 1f32 } else { 0f32 });
            self.constraints.push(t.constraint);
            self.next_constraints.push(t.next_constraints);
            self.extras.push(t.extra);
        } else {
            self.obss[self.head] = t.obs;
            self.actions[self.head] = t.action;
            self.rewards[self.head] = t.reward;
            self.next_obss[self.head] = t.next_obs;
            self.terminated[self.head] = if t.terminated { 1f32 } else { 0f32 };
            self.truncated[self.head] = if t.truncated { 1f32 } else { 0f32 };
            self.constraints[self.head] = t.constraint;
            self.next_constraints[self.head] = t.next_constraints;
            self.extras[self.head] = t.extra;
        }

        self.head = (self.head + 1) % self.capacity;
    }

    pub fn len(&self) -> usize { self.obss.len() }

    pub fn sample(&mut self, batch_size: usize) -> Option<Batch<Obs, Action, Constraint, Extra>> {
        let len = self.len();

        if len < batch_size { return None; }

        let indices: Vec<usize> = (0..batch_size).map(|_| self.rng.random_range(0..len)).collect();

        let (o, a, r, no, te, tr, b, nb, ex): (Vec<Obs>, Vec<Action>, Vec<f32>, Vec<Obs>, Vec<f32>, Vec<f32>, Vec<Constraint>, Vec<Constraint>, Vec<Extra>)
            = indices.iter().map(|&index| {(
                    self.obss[index].clone(),
                    self.actions[index].clone(),
                    self.rewards[index],
                    self.next_obss[index].clone(),
                    self.terminated[index],
                    self.truncated[index],
                    self.constraints[index].clone(),
                    self.next_constraints[index].clone(),
                    self.extras[index].clone())
            }).collect();
        Some(Batch {
            obss: Obs::batch(o, &self.device),
            actions: Action::batch(a, &self.device),
            rewards: Tensor::from_floats(r.as_slice(), &self.device),
            next_obss: Obs::batch(no, &self.device),
            terminated: Tensor::from_floats(te.as_slice(), &self.device),
            truncated: Tensor::from_floats(tr.as_slice(), &self.device),
            constraints: Constraint::batch(b, &self.device),
            next_constraints: Constraint::batch(nb, &self.device),
            extras: Extra::batch(ex, &self.device),

            batch_size,
        })
    }
}