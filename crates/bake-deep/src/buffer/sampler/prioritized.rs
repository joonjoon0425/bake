//! A sampler which uses Priority (PER)
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use burn::prelude::*;
use crate::{buffer::{replay::LazyStorage, sampler::{SampleInfo, Sampler, SamplerConfig}}, data::{Batch, Batchable}};

/// A sampler which uses Priority (PER)
pub struct PrioritizedSampler {
    alpha: f64,
    beta: f64,

    sum_tree: SumTree,
    min_tree: MinTree,
    priority_clip: Option<f64>,
    /// holds the max_priority value
    max_priority: f64,

    eps: f64,
}

impl PrioritizedSampler {
    /// create a new PrioritizedSampler
    pub fn new(seed: u64, alpha: f64, beta: f64, capacity: usize, priority_clip: Option<f64>) -> Self {
        Self {
            alpha,
            beta,
            priority_clip,
            max_priority: 1.,
            eps: 1e-6,
            sum_tree: SumTree::new(seed, capacity),
            min_tree: MinTree::new(capacity),
        }
    }

    /// return the beta
    pub fn beta(&self) -> f64 { self.beta }
    /// return the mutable reference of beta
    pub fn beta_mut(&mut self) -> &mut f64 { &mut self.beta }
}

impl Sampler for PrioritizedSampler {
    fn after_push(&mut self, index: usize) {
        self.sum_tree.update(index, self.max_priority);
        self.min_tree.update(index, self.max_priority);
    }

    fn sample<Obs, Action, Constraint>(&mut self, sample_size: usize, storage: &LazyStorage<Obs, Action, Constraint>) -> (Batch<Obs, Action, Constraint>, SampleInfo)
    where
        Obs: Batchable,
        Action: Batchable,
        Constraint: Batchable,
    {
        let device = storage.device();
        let indices_raw = self.sum_tree.sample_idx(sample_size);
        let indices = Tensor::from_ints(indices_raw.as_slice(), &device);
        let total = self.sum_tree.sum();
        let p_min = self.min_tree.min() / total;
        let max_w = (p_min as f64).powf(-self.beta);
        let is_weights: Vec<f32> = indices_raw.iter().map(
            |&i| {
                let p = self.sum_tree.get(i) / total;
                (((p as f64).powf(-self.beta)) / max_w) as f32
            }
        ).collect();
        let is_weights = Tensor::from_floats(is_weights.as_slice(), &device);
        let selected = storage.select(indices);
        (selected, SampleInfo { indices: indices_raw, is_weights: Some(is_weights) })
    }
}

impl PrioritizedSampler {
    /// update the priority from given indices and priorites
    pub fn update_priority(&mut self, indices: &[usize], priorities: Tensor<1>) {
        let e = match self.priority_clip {
            Some(c) => priorities.abs().clamp_max(c as f32),
            None => priorities.abs()
        };
        let p: Vec<f32> = (e + self.eps).powf_scalar(self.alpha).into_data().try_into_vec().unwrap();
        // update the priority
        for (i, &index) in indices.iter().enumerate() {
            let v = p[i] as f64;
            self.sum_tree.update(index, v);
            self.min_tree.update(index, v);
            self.max_priority = self.max_priority.max(v);
        }
    }
}


/// Configuration for PrioritizedSampler
pub struct PrioritizedSamplerConfig {
    /// controls how much the sampler will care about priorities. 0 -> uniform, 1 -> fully prioritized
    pub alpha: f64,
    /// controls the importance sampling weights. 0 -> no effect, 1 -> full correction
    pub beta: f64,
    /// clip the maximum priority. default None
    pub priority_clip: Option<f64>
}

impl PrioritizedSamplerConfig {
    /// create a new PrioritizedSamplerConfig
    /// - `priority_clip` is `None` by default
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha,
            beta,
            priority_clip: None
        }
    }

    /// configure the priority clip
    pub fn with_priority_clip(mut self, priority_clip: f64) -> Self {
        self.priority_clip = Some(priority_clip);
        self
    }
}

impl SamplerConfig for PrioritizedSamplerConfig {
    type SamplerType = PrioritizedSampler;
    fn init(self, seed: u64, capacity: usize) -> Self::SamplerType {
        PrioritizedSampler::new(seed, self.alpha, self.beta, capacity, self.priority_clip)
    }
}

#[derive(Debug, Clone)]
struct SumTree {
    tree: Vec<f64>,
    rng: SmallRng,
    n: usize,
}

impl SumTree {
    pub fn new(seed: u64, capacity: usize) -> Self {
        let depth = (capacity as f64).log2().ceil();
        let n = depth.exp2() as usize;
        let tree = vec![0f64; 2 * n];
        Self { tree, n, rng: SmallRng::seed_from_u64(seed) }
    }

    #[cfg(test)]
    fn from_vec(seed: u64, vec: Vec<f64>) -> Self {
        let capacity = vec.len();
        let depth = (capacity as f64).log2().ceil();
        let n = depth.exp2() as usize;
        let mut tree = vec![0f64; 2 * n];
        for i in 0..capacity {
            tree[n + i] = vec[i];
        }
        let mut index = n / 2;
        while index >= 1 {
            let mut tmp = index;
            let r = index * 2 - 1;
            while tmp <= r {
                tree[tmp] = tree[2 * tmp] + tree[2 * tmp + 1];
                tmp += 1;
            }
            index /= 2;
        }

        Self { tree, n, rng: SmallRng::seed_from_u64(seed) }
    }

    pub fn update(&mut self, index: usize, val: f64) {
        let mut index = index + self.n;
        self.tree[index] = val;

        index /= 2;
        while index >= 1 {
            self.tree[index] = self.tree[2 * index] + self.tree[2 * index + 1];
            index /= 2;
        }
    }

    pub fn sum(&self) -> f64 {
        self.tree[1]
    }

    pub fn get(&self, index: usize) -> f64 {
        self.tree[index + self.n]
    }

    pub fn sample_idx(&mut self, n: usize) -> Vec<usize> {
        let range = self.sum() / (n as f64);
        let mut vec = vec![0usize; n];
        for i in 0..n {
            let mut r = self.rng.random_range(i as f64 * range..(i + 1) as f64 * range);
            let mut index = 1;
            while index < self.n {
                if self.tree[2 * index] >= r {
                    index = 2 * index;
                } else {
                    r -= self.tree[2 * index];
                    index = 2 * index + 1;
                }
            }
            vec[i] = index - self.n;
        }

        vec
    }

    // #[cfg(test)]
    // fn is_correct(&self) {
    //     let mut index = self.n / 2;
    //     while index >= 1 {
    //         let mut tmp = index;
    //         let r = index * 2 - 1;
    //         while tmp <= r {
    //             if self.tree[tmp] != self.tree[2 * tmp] + self.tree[2 * tmp + 1] {
    //                 panic!("Wrong sum on index {tmp}");
    //             }
    //             tmp += 1;
    //         }
    //         index /= 2;
    //     }
    // }
}

#[derive(Debug, Clone)]
struct MinTree { tree: Vec<f64>, n: usize }

impl MinTree {
    pub fn new(capacity: usize) -> Self {
        let n = (capacity as f64).log2().ceil().exp2() as usize;
        Self { tree: vec![f64::INFINITY; 2 * n], n }
    }
    pub fn update(&mut self, index: usize, val: f64) {
        let mut i = index + self.n;
        self.tree[i] = val;
        i /= 2;
        while i >= 1 {
            self.tree[i] = self.tree[2 * i].min(self.tree[2 * i + 1]);
            i /= 2;
        }
    }
    pub fn min(&self) -> f64 { self.tree[1] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_distribution_converges() {
        let priors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut tree = SumTree::from_vec(12, priors.clone());
        let mut counts = [0usize; 8];
        for _ in 0..60_000 {
            for i in tree.sample_idx(8) { counts[i] += 1; }
        }
        let total: usize = counts.iter().sum();
        let sum: f64 = priors.iter().sum();
        for i in 0..8 {
            let observed = counts[i] as f64 / total as f64;
            let expected = priors[i] / sum;
            assert!((observed - expected).abs() < 5e-3, "leaf {i}: {observed} vs {expected}");
        }
    }
}