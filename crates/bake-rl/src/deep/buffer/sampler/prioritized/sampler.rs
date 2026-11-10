use super::sum_tree::SumTree;
use super::min_tree::MinTree;
use super::super::{Sampler, Sample};

use crate::core::data::{BatchOps, Shaped};

use burn::prelude::*;

/// samples using priorities
/// - used for prioritized replay buffer
pub struct PrioritizedSampler {
    alpha: f32,
    beta: f32,

    sum_tree: SumTree,
    min_tree: MinTree,

    priority_clip: Option<f32>,

    max_priority: f32,

    eps: f32,
}

impl PrioritizedSampler {
    /// create new `PrioritizedSampler`
    pub fn new(alpha: f32, beta: f32, capacity: usize, priority_clip: Option<f32>, device: &Device) -> Self {
        Self {
            alpha,
            beta,
            priority_clip,
            max_priority: 1.,
            eps: 1e-6,
            sum_tree: SumTree::new(capacity as u64, device),
            min_tree: MinTree::new(capacity as u64, device)
        }
    }

    /// return `beta`
    pub fn beta(&self) -> f32 { self.beta }

    /// return mutable reference of `beta`
    pub fn beta_mut(&mut self) -> &mut f32 { &mut self.beta }

    /// update priorities
    pub fn update(&mut self, indices: Tensor<1, Int>, priorities: Tensor<1>) {
        let e = match self.priority_clip {
            Some(c) => priorities.abs().clamp_max(c),
            None => priorities.abs(),
        };
        let p = (e + self.eps).powf_scalar(self.alpha);

        self.sum_tree.update(indices.clone(), p.clone());
        self.min_tree.update(indices, p);
    }
}

impl Sampler for PrioritizedSampler {
    /// importance sampling weights
    type SampleInfo = Tensor<1>;

    fn sample<T: BatchOps>(&self, size: usize, _len: usize, storage: &Shaped<T>) -> Sample<T, Self::SampleInfo> {
        let (priorities, indices) = self.sum_tree.sample_stratified(size as i64);
        let total = self.sum_tree.total_sum();
        let p_min = self.min_tree.min() / total.clone();
        let max_w = p_min.powf_scalar(-self.beta);
        let is_weights = (priorities / total).powf_scalar(-self.beta) / max_w;

        let sample = storage.clone().select(0, indices.clone());
        Sample {
            sample,
            indices,
            info: is_weights,
        }
    }

    fn after_push(&mut self, indices: Tensor<1, Int>) {
        let values = Tensor::full_like(&indices, self.max_priority).float();
        self.sum_tree.update(indices.clone(), values.clone());
        self.min_tree.update(indices, values);
    }
}