use burn::{Tensor, tensor::{Bool, Int}};

use crate::constraint::{DiscreteMask, Unconstrained};

pub trait Indexable {
    fn select(&self, idx: Tensor<1, Int>) -> Self;
}

impl<const D: usize> Indexable for Tensor<D> {
    fn select(&self, idx: Tensor<1, Int>) -> Self {
        self.clone().select(0, idx)
    }
}

impl<const D: usize> Indexable for Tensor<D, Int> {
    fn select(&self, idx: Tensor<1, Int>) -> Self {
        self.clone().select(0, idx)
    }
}

impl<const D: usize> Indexable for Tensor<D, Bool> {
    fn select(&self, idx: Tensor<1, Int>) -> Self {
        self.clone().select(0, idx)
    }
}

impl<const D: usize> Indexable for DiscreteMask<D> {
    fn select(&self, idx: Tensor<1, Int>) -> Self {
        DiscreteMask(self.0.clone().select(0, idx))
    }
}

impl Indexable for () {
    fn select(&self, _: Tensor<1, Int>) -> Self {
        ()
    }
}

impl Indexable for Unconstrained {
    fn select(&self, _: Tensor<1, Int>) -> Self {
        Unconstrained
    }
}