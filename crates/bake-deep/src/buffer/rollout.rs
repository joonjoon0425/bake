//! Rollout buffer for on-policy methods
//! 
use burn::prelude::*;
use crate::data::{Batch, Batchable};
/// Rollout buffer implementation.
pub struct RolloutBuffer<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable = ()> {
    batch: Vec<Batch<Obs, Action, Constraint, Extra>>,
    device: Device,
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable> RolloutBuffer<Obs, Action, Constraint, Extra> {
    /// create a new `RolloutBuffer`
    pub fn new(mut device: Device) -> Self {
        if !device.is_autodiff() { device = device.autodiff(); }
        Self { batch: vec![], device }
    }

    /// return the current length of buffer
    pub fn len(&self) -> usize {
        self.batch.len()
    }

    /// push one transition into buffer
    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Extra>) {
        self.batch.push(t);
    }

    /// pop all elements of buffer
    pub fn pop(&mut self) -> Batch<Obs, Action, Constraint, Extra> {
        let batch = std::mem::replace(&mut self.batch, vec![]);
        Batch::cat(batch).to_device(&self.device)
    }
}

#[cfg(test)]
mod tests {
    use burn::{prelude::*, tensor::Distribution};
    use crate::{buffer::rollout::RolloutBuffer, constraint::Unconstrained, data::{Batch, Batchable}};

    #[test]
    fn init_test() {
        let device = Device::default();
        let mut buffer = RolloutBuffer::new(device.clone());

        let obs = Tensor::<2>::full([1, 4], 1000.0, &device);
        let action = Tensor::<1, Int>::random([1], Distribution::Uniform(0.0, 2.0), &device);
        let reward = Tensor::<1>::from_floats([1.0], &device);
        let batch = Batch {
            obss: obs.clone(),
            actions: action.clone(),
            rewards: reward.clone(),
            next_obss: obs.clone(),
            constraints: Unconstrained,
            next_constraints: Unconstrained,
            terminated: reward.clone(),
            truncated: reward.clone(),
            extras: (),
        };
        buffer.push(batch);
    }

    #[test]
    fn pop_test() {
        let device = Device::default();
        let mut buffer = RolloutBuffer::new(device.clone());

        for i in 0..11 {
            let obs = Tensor::<2>::full([1, 4], i as f32, &device);
            let action = Tensor::<1, Int>::random([1], Distribution::Uniform(0.0, 2.0), &device);
            let reward = Tensor::<1>::from_floats([1.0], &device);
            let batch = Batch {
                obss: obs.clone(),
                actions: action.clone(),
                rewards: reward.clone(),
                next_obss: obs.clone(),
                constraints: Unconstrained,
                next_constraints: Unconstrained,
                terminated: reward.clone(),
                truncated: reward.clone(),
                extras: (),
            };
            buffer.push(batch);
        }

        let rollout = buffer.pop();
        assert!(rollout.len().unwrap() == 11);
    }
}