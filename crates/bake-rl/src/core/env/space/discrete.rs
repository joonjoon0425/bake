//! Space for discrete elements

use burn::prelude::*;
use rand::prelude::*;

use super::{NativeSpace, BurnSpace, ToBurnSpace};

/// Discrete space
#[derive(Debug, Clone)]
pub struct DiscreteSpace<Repr: DiscreteSpaceRepr> {
    n: usize,
    inner: Repr::Inner,
}

/// Discrete Space Representation
pub trait DiscreteSpaceRepr {
    /// inner types which each representation requires
    type Inner;
}

impl DiscreteSpaceRepr for crate::core::Native { type Inner = (); }
impl DiscreteSpaceRepr for crate::core::Burn { type Inner = Device; }

impl DiscreteSpace<crate::core::Burn> {
    /// create a new `DiscreteSpace` with `Burn` representation
    pub fn burn(n: usize, device: &Device) -> Self {
        Self { n, inner: device.clone() }
    }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
}

impl BurnSpace for DiscreteSpace<crate::core::Burn> {
    type Element = Tensor<1, Int>;

    fn sample(&self, n: usize) -> Self::Element {
        Tensor::random([n], burn::tensor::Distribution::Uniform(0., self.n as f64), &self.inner)
    }
    
    fn device(&self) -> Device {
        self.inner.clone()
    }
}

impl DiscreteSpace<crate::core::Native> {
    /// create a new `DiscreteSpace` with `Native` representation
    pub fn native(n: usize) -> Self {
        Self { n, inner: () }
    }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
}

impl NativeSpace for DiscreteSpace<crate::core::Native> {
    type Element = usize;

    fn sample<R: Rng>(&self, mut rng: R) -> Self::Element {
        rng.random_range(0..self.n)
    }
}

impl ToBurnSpace for DiscreteSpace<crate::core::Native> {
    type BurnSpace = DiscreteSpace<crate::core::Burn>;

    fn into_burn(self, device: &Device) -> Self::BurnSpace {
        Self::BurnSpace::burn(self.n, device)
    }

    fn bind(elems: Vec<Self::Element>, device: &Device) -> <Self::BurnSpace as BurnSpace>::Element {
        assert!(!elems.is_empty(), "ToBurnSpace::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        Tensor::from_ints(&elems[..], device)
    }

    fn unbind(elem: <Self::BurnSpace as BurnSpace>::Element) -> Vec<Self::Element> {
        let native = elem.into_data().try_into_vec_as::<u64>().unwrap();
        native.into_iter().map(|u| u as usize).collect()
    }
}