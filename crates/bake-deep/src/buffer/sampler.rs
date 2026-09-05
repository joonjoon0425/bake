//! A sampler trait and implementations for buffers
//! 

use rand::{RngExt, SeedableRng, rngs::SmallRng};
use burn::prelude::*;
use crate::data::{Batch, Batchable};

/// A `Sampler` trait which all samplers for buffers must implement
pub trait Sampler {
    /// sample n elements from given storage (currently only Batch type)
    fn sample<Obs, Action, Constraint, Extra>(&mut self, n: usize, storage: &Batch<Obs, Action, Constraint, Extra>) -> (Batch<Obs, Action, Constraint, Extra>, SampleInfo)
    where
        Obs: Batchable,
        Action: Batchable,
        Constraint: Batchable,
        Extra: Batchable;
}

/// A struct holding the information of samples
pub struct SampleInfo {
    /// the indices of sample as a member of given buffer
    pub indices: Vec<usize>
}

/// A basic sampler which treats all elements equally
pub struct UniformSampler {
    rng: SmallRng,
}

impl UniformSampler {
    /// create a new `UniformSampler`
    pub fn new(seed: u64) -> Self { Self { rng: SmallRng::seed_from_u64(seed) } }
}

impl Sampler for UniformSampler {
    /// # Panic
    /// panics when the length of the storage is smaller than n
    fn sample<Obs, Action, Constraint, Extra>(&mut self, n: usize, storage: &Batch<Obs, Action, Constraint, Extra>) -> (Batch<Obs, Action, Constraint, Extra>, SampleInfo)
    where
        Obs: Batchable,
        Action: Batchable,
        Constraint: Batchable,
        Extra: Batchable
    {
        let len = storage.len().unwrap();
        if len < n { panic!("Sampler received n bigger than given storage's length") }

        let indices_raw: Vec<usize> = (0..n).map(|_| self.rng.random_range(0..len)).collect();
        let indices = Tensor::from_ints(indices_raw.as_slice(), &storage.device());
        (storage.clone().select(indices), SampleInfo { indices: indices_raw } )
    }
}

/// A sampler which uses Priority (PER)
pub struct PrioritizedSampler {
    alpha: f64,
    beta: f64,

    sum_tree: SumTree,
    min_tree: MinTree,
    priority_clip: Option<f64>,
    max_priority: f64,
}

impl PrioritizedSampler {
    pub fn new(seed: u64, alpha: f64, beta: f64, capacity: usize, priority_clip: Option<f64>, max_priority: f64) -> Self {
        Self {
            alpha,
            beta,
            priority_clip,
            max_priority,
            sum_tree: SumTree::new(seed, capacity),
            min_tree: MinTree::new(capacity),
        }
    }
}

/// Configuration for PrioritizedSampler
pub struct PrioritizedSamplerConfig {
    pub alpha: f64,
    pub beta: f64,
    pub seed: u64,
    pub capacity: usize,
    /// default None
    pub priority_clip: Option<f64>,
    /// default 1e-6
    pub max_priority: f64,
}

impl PrioritizedSamplerConfig {
    /// create a new PrioritizedSamplerConfig
    /// - `priority_clip` is `None` by default
    /// - `max_priority` is 1e-6 by default
    pub fn new(seed: u64, alpha: f64, beta: f64, capacity: usize) -> Self {
        Self {
            seed,
            alpha,
            beta,
            capacity,
            priority_clip: None,
            max_priority: 1e-6
        }
    }

    /// configure the priority clip
    pub fn with_priority_clip(mut self, priority_clip: f64) -> Self {
        self.priority_clip = Some(priority_clip);
        self
    }

    /// configure the max priority for newly pushed samples
    pub fn with_max_priority(mut self, max_priority: f64) -> Self {
        self.max_priority = max_priority;
        self
    }

    /// create a new `PrioritizedSampler` from configuration
    pub fn init(self) -> PrioritizedSampler {
        PrioritizedSampler::new(self.seed, self.alpha, self.beta, self.capacity, self.priority_clip, self.max_priority)
    }
}

impl Sampler for PrioritizedSampler {
    fn sample<Obs, Action, Constraint, Extra>(&mut self, n: usize, storage: &Batch<Obs, Action, Constraint, Extra>) -> (Batch<Obs, Action, Constraint, Extra>, SampleInfo)
    where
        Obs: Batchable,
        Action: Batchable,
        Constraint: Batchable,
        Extra: Batchable
    {
        let len = storage.len().unwrap();
        let device = storage.device();
        let indices = self.sum_tree.sample_idx(n);
        let total = self.sum_tree.sum();
        let p_min = self.min_tree.min() / total;
        let max_w = (p_min * len as f64).powf(-self.beta);
        let is_weights: Vec<f32> = indices.iter().map(
            |&i| {
                let p = self.sum_tree.get(i) / total;
                (((p * len as f64).powf(-self.beta)) / max_w) as f32
            }
        ).collect();
        let is_weights = Tensor::from_floats(is_weights.as_slice(), &device);
        let selected = 
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

    pub fn from_vec(seed: u64, vec: Vec<f64>) -> Self {
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

    #[cfg(test)]
    pub fn is_correct(&self) {
        let mut index = self.n / 2;
        while index >= 1 {
            let mut tmp = index;
            let r = index * 2 - 1;
            while tmp <= r {
                if self.tree[tmp] != self.tree[2 * tmp] + self.tree[2 * tmp + 1] {
                    panic!("Wrong sum on index {tmp}");
                }
                tmp += 1;
            }
            index /= 2;
        }
    }
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