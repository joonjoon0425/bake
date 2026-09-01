//! A priortized experience replay buffer implementation
//! 
use burn::Tensor;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::types::{Batch, Batchable};
pub struct PrioritizedExperienceReplayBuffer<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable = ()> {
    capacity: usize,
    head: usize,

    alpha: f64,
    beta: f64,

    batch: Vec<Batch<Obs, Action, Constraint, Extra>>,
    sum_tree: SumTree,
    min_tree: MinTree,
    priority_clip: Option<f64>,
    max_priority: f64,
}

impl<Obs, Action, Constraint, Extra> PrioritizedExperienceReplayBuffer<Obs, Action, Constraint, Extra>
where
    Obs: Batchable,
    Action: Batchable,
    Constraint: Batchable,
    Extra: Batchable
{
    pub fn new(seed: u64, capacity: usize, alpha: f64, beta: f64, priority_clip: Option<f64>) -> Self {
        let batch = Vec::with_capacity(capacity);
        let sum_tree = SumTree::new(seed, capacity);
        let min_tree = MinTree::new(capacity);
        Self {
            batch,
            sum_tree,
            min_tree,
            head: 0,
            capacity,
            alpha,
            beta,
            max_priority: 1f64,
            priority_clip,
        }
    }

    pub fn len(&self) -> usize { self.batch.len() }

    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Extra>) {
        if self.len() < self.capacity {
            self.batch.push(t);    
        } else {
            self.batch[self.head] = t;
        }
        self.sum_tree.update(self.head, self.max_priority);
        self.min_tree.update(self.head, self.max_priority);
        self.head = (self.head + 1) % self.capacity;
    }

    pub fn update(&mut self, indices: &[usize], td_errors: Tensor<1>) {
        let e = match self.priority_clip {
            Some(c) => td_errors.abs().clamp_max(c as f32),
            None => td_errors.abs()
        };
        let p: Vec<f32> = (e + 1e-6).powf_scalar(self.alpha).into_data().try_into_vec().unwrap();
        for (i, &index) in indices.iter().enumerate() {
            let v = p[i] as f64;
            self.sum_tree.update(index, v);
            self.min_tree.update(index, v);
            self.max_priority = self.max_priority.max(v);
        }
    }

    pub fn sample(&mut self, batch_size: usize) -> Option<(Batch<Obs, Action, Constraint, Tensor<1>>, Vec<usize>)> {
        let len = self.len();

        if len < batch_size { return None; }

        let indices = self.sum_tree.sample_idx(batch_size);
        let total = self.sum_tree.sum();
        let p_min = self.min_tree.min() / total;
        let max_w = (p_min * len as f64).powf(-self.beta);
        let is_weights: Vec<f32> = indices.iter().map(
            |&i| {
                let p = self.sum_tree.get(i) / total;
                (((p * len as f64).powf(-self.beta)) / max_w) as f32
        }).collect();
        let is_weights = Tensor::from_floats(is_weights.as_slice(), &self.batch[0].device());
        let selected: Vec<Batch<Obs, Action, Constraint, Extra>> = indices.iter().map(|&i| self.batch[i].clone()).collect();
        let selected = Batch::concat(selected).add_extra(is_weights);
        Some((selected, indices))
    }

    pub fn beta(&self) -> f64 { self.beta }
    pub fn beta_mut(&mut self) -> &mut f64 { &mut self.beta }

}

#[derive(Debug, Clone)]
pub struct SumTree {
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
pub struct MinTree { tree: Vec<f64>, n: usize }

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