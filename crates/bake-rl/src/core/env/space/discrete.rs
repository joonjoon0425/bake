//! Space for discrete elements

use burn::prelude::*;
use rand::prelude::*;

use super::{Space, BatchedSpace, IntoBatchedSpace};

/// Discrete space
#[derive(Debug, Clone)]
pub struct DiscreteSpace {
    n: usize,
}

impl DiscreteSpace {
    /// create a new `DiscreteSpace`
    pub fn new(n: usize) -> Self {
        Self { n }
    }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
}

impl Space for DiscreteSpace {
    type Element = usize;

    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Element {
        rng.random_range(0..self.n)
    }
}

/// Discrete space
#[derive(Debug, Clone)]
pub struct BatchedDiscreteSpace {
    n: usize,
    n_envs: usize,
    device: Device,
}

impl BatchedDiscreteSpace {
    /// create a new `BatchedDiscreteSpace`
    pub fn new(n: usize, n_envs: usize, device: &Device) -> Self {
        Self { n, n_envs, device: device.clone() }
    }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
}

impl BatchedSpace for BatchedDiscreteSpace {
    type Element = Tensor<1, Int>;

    fn sample(&self) -> Self::Element {
        Tensor::random([self.n_envs], burn::tensor::Distribution::Uniform(0., self.n as f64), &self.device)
    }
    
    fn device(&self) -> Device {
        self.device.clone()
    }
}

impl IntoBatchedSpace for DiscreteSpace {
    type Batched = BatchedDiscreteSpace;

    fn into_batched(spaces: &[Self], device: &Device) -> Self::Batched {
        assert!(spaces.is_empty(), "IntoBatchedSpace::into_batched: got empty space vector");
        let n = spaces[0].n;
        assert!(spaces.iter().all(|s|s.n == n), "IntoBatchedSpace::into_batched: the number of elements of each `DiscreteSpace` is not same");
        Self::Batched::new(n, spaces.len(), device)
    }

    fn bind(elems: Vec<Self::Element>, device: &Device) -> <Self::Batched as BatchedSpace>::Element {
        assert!(!elems.is_empty(), "ToBatchedSpace::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        Tensor::from_ints(&elems[..], device)
    }

    fn unbind(elem: <Self::Batched as BatchedSpace>::Element) -> Vec<Self::Element> {
        let native = elem.into_data().try_into_vec_as::<u64>().unwrap();
        native.into_iter().map(|u| u as usize).collect()
    }
}