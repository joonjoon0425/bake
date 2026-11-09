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
    device: Device,
}

impl BatchedDiscreteSpace {
    /// create a new `BatchedDiscreteSpace`
    pub fn new(n: usize, device: &Device) -> Self {
        Self { n, device: device.clone() }
    }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
}

impl BatchedSpace for BatchedDiscreteSpace {
    type Element = Tensor<1, Int>;

    fn sample(&self, n: usize) -> Self::Element {
        Tensor::random([n], burn::tensor::Distribution::Uniform(0., self.n as f64), &self.device)
    }
    
    fn device(&self) -> Device {
        self.device.clone()
    }
}

impl IntoBatchedSpace for DiscreteSpace {
    type Batched = BatchedDiscreteSpace;

    fn into_batched(self, device: &Device) -> Self::Batched {
        Self::Batched::new(self.n, device)
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