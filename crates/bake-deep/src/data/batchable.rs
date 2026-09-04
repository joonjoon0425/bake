//! A Batchable trait

use core::ops::Range;
use std::ops::IndexMut;

use burn::{
    Tensor, tensor::{Bool, Int, Device},
};

use crate::constraint::{discrete_constraint::DiscreteMask, Unconstrained};

/// Can create into batch along the batch dimension
pub trait Batchable: std::fmt::Debug + Sized + Clone + Send + Sync + 'static {
    /// returns the length of the batch. returns `None` if there are no information of length (`Unconstrained` or `()`).
    fn len(&self) -> Option<usize>;

    /// concatanate along batch dimension (dim 0)
    ///
    /// # Panics
    /// empty `items` panics
    fn cat(items: Vec<Self>) -> Self;

    /// gather elements of batches from indices `idx`. The ordering may change.
    fn select(self, idx: Tensor<1, Int>) -> Self;

    /// gather elements of batches from `range`. The ordering does not changes.
    fn slice(self, range: Range<usize>) -> Self;

    /// detach from the autodiff graph, if possible.
    fn detach(self) -> Self {
        self
    }

    /// checks if the batch is empty. If the batch has no length information (`Unconstraind` or `()`), it returns false
    fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }

    /// assign in-place the given batch, starting from the given index of batch dimension (dim 0)
    fn assign_inplace(&mut self, data: Self, index: usize);

    /// make zeros with given capacity, and shape
    fn zeros_like(capacity: usize, data: &Self, device: &Device) -> Self;

    /// move all data to given device
    fn to_device(self, device: &Device) -> Self;
}

// implementations
impl<const D: usize> Batchable for Tensor<D> {
    fn len(&self) -> Option<usize> {
        Some(self.shape()[0])
    }

    fn cat(items: Vec<Self>) -> Self {
        assert!(!items.is_empty(), "Batchable::cat on an empty Vec");
        Tensor::cat(items, 0)
    }

    fn select(self, idx: Tensor<1, Int>) -> Self {
        self.select(0, idx)
    }

    fn slice(self, range: Range<usize>) -> Self {
        self.narrow(0, range.start, range.len())
    }

    // detach only exists on float types
    fn detach(self) -> Self {
        self.detach()
    }

    fn assign_inplace(&mut self, data: Self, index: usize) {
        let len = data.len().unwrap();
        self.inplace(|a| a.slice_assign(index..index + len, data));
    }

    fn zeros_like(capacity: usize, data: &Self, device: &Device) -> Self {
        let mut shape = data.shape();
        *shape.index_mut(0) = capacity;
        Tensor::zeros(shape, device)
    }

    fn to_device(self, device: &Device) -> Self {
        self.to_device(device)
    }
}

impl<const D: usize> Batchable for Tensor<D, Int> {
    fn len(&self) -> Option<usize> {
        Some(self.shape()[0])
    }

    fn cat(items: Vec<Self>) -> Self {
        assert!(!items.is_empty(), "Batchable::cat on an empty Vec");
        Tensor::cat(items, 0)
    }

    fn select(self, idx: Tensor<1, Int>) -> Self {
        self.select(0, idx)
    }

    fn slice(self, range: Range<usize>) -> Self {
        self.narrow(0, range.start, range.len())
    }

    fn assign_inplace(&mut self, data: Self, index: usize) {
        let len = data.len().unwrap();
        self.inplace(|a| a.slice_assign(index..index + len, data));
    }

    fn zeros_like(capacity: usize, data: &Self, device: &Device) -> Self {
        let mut shape = data.shape();
        *shape.index_mut(0) = capacity;
        Tensor::zeros(shape, device)
    }

    fn to_device(self, device: &Device) -> Self {
        self.to_device(device)
    }
}

impl<const D: usize> Batchable for Tensor<D, Bool> {
    fn len(&self) -> Option<usize> {
        Some(self.shape()[0])
    }

    fn cat(items: Vec<Self>) -> Self {
        assert!(!items.is_empty(), "Batchable::cat on an empty Vec");
        Tensor::cat(items, 0)
    }

    fn select(self, idx: Tensor<1, Int>) -> Self {
        self.select(0, idx)
    }

    fn slice(self, range: Range<usize>) -> Self {
        self.narrow(0, range.start, range.len())
    }

    fn assign_inplace(&mut self, data: Self, index: usize) {
        let len = data.len().unwrap();
        self.inplace(|a| a.slice_assign(index..index + len, data));
    }

    fn zeros_like(capacity: usize, data: &Self, device: &Device) -> Self {
        let mut shape = data.shape();
        *shape.index_mut(0) = capacity;
        Tensor::zeros(shape, device)
    }

    fn to_device(self, device: &Device) -> Self {
        self.to_device(device)
    }
}

impl<const D: usize> Batchable for DiscreteMask<D> {
    fn len(&self) -> Option<usize> {
        Some(self.0.shape()[0])
    }

    fn cat(items: Vec<Self>) -> Self {
        assert!(!items.is_empty(), "Batchable::cat on an empty Vec");
        let inner: Vec<Tensor<D, Bool>> = items.into_iter().map(|t| t.0).collect();
        DiscreteMask(Tensor::cat(inner, 0))
    }

    fn select(self, idx: Tensor<1, Int>) -> Self {
        DiscreteMask(self.0.select(0, idx))
    }

    fn slice(self, range: Range<usize>) -> Self {
        DiscreteMask(self.0.narrow(0, range.start, range.len()))
    }

    fn assign_inplace(&mut self, data: Self, index: usize) {
        let len = data.len().unwrap();
        self.0.inplace(|a| a.slice_assign(index..index + len, data.0));
    }

    fn zeros_like(capacity: usize, data: &Self, device: &Device) -> Self {
        let mut shape = data.0.shape();
        *shape.index_mut(0) = capacity;
        DiscreteMask(Tensor::zeros(shape, device))
    }

    fn to_device(self, device: &Device) -> Self {
        DiscreteMask(self.0.to_device(device))
    }
}

impl Batchable for Unconstrained {
    fn len(&self) -> Option<usize> {
        None
    }

    fn cat(items: Vec<Self>) -> Self {
        assert!(!items.is_empty(), "Batchable::cat on an empty Vec");
        Unconstrained
    }

    fn select(self, _: Tensor<1, Int>) -> Self {
        Unconstrained
    }

    fn slice(self, _: Range<usize>) -> Self {
        Unconstrained
    }

    fn assign_inplace(&mut self, _: Self, _: usize) {}

    fn zeros_like(_: usize, _: &Self, _: &Device) -> Self { Unconstrained }

    fn to_device(self, _: &Device) -> Self { Unconstrained }
}

impl Batchable for () {
    fn len(&self) -> Option<usize> {
        None
    }

    fn cat(items: Vec<Self>) -> Self {
        assert!(!items.is_empty(), "Batchable::cat on an empty Vec");
    }

    fn select(self, _: Tensor<1, Int>) -> Self {}

    fn slice(self, _: Range<usize>) -> Self {}

    fn assign_inplace(&mut self, _: Self, _: usize) {}

    fn zeros_like(_: usize, _: &Self, _: &Device) -> Self {}

    fn to_device(self, _: &Device) -> Self {}
}

impl<T: Batchable> Batchable for Option<T> {
    fn len(&self) -> Option<usize> {
        self.as_ref().and_then(Batchable::len)
    }

    /// The elements must be all `Some` or all `None`. Cannot be mixed up.
    ///
    /// # Panics
    /// empty `items` or mixed inhomogeneous values (`Some` and `None`) panics
    fn cat(items: Vec<Self>) -> Self {
        assert!(!items.is_empty(), "Batchable::cat on an empty Vec");
        let n_some = items.iter().filter(|v| v.is_some()).count();
        if n_some == 0 {
            return None;
        }
        assert_eq!(
            n_some,
            items.len(),
            "Batchable::cat on Option: cannot mix Some and None ({} Some of {})",
            n_some,
            items.len()
        );
        Some(T::cat(items.into_iter().map(|v| v.unwrap()).collect()))
    }

    fn select(self, idx: Tensor<1, Int>) -> Self {
        self.map(|v| v.select(idx))
    }

    fn slice(self, range: Range<usize>) -> Self {
        self.map(|v| v.slice(range))
    }

    fn detach(self) -> Self {
        self.map(Batchable::detach)
    }

    /// The elements must be all `Some` or all `None`. Cannot be mixed up.
    ///
    /// # Panics
    /// mixed values (`Some` and `None`) panics
    fn assign_inplace(&mut self, data: Self, index: usize) {
        match (self, data) {
            (Some(dst), Some(src)) => {
                dst.assign_inplace(src, index);
            },
            (None, None) => {},
            _ => {
                panic!("optional self and optional data must be: both Some or both None.");
            }
        }
    }

    fn zeros_like(capacity: usize, data: &Self, device: &Device) -> Self {
        match data {
            Some(v) => {
                Some(T::zeros_like(capacity, v, device))
            },
            None => {
                None
            }
        }
    }

    fn to_device(self, device: &Device) -> Self {
        self.map(|v| v.to_device(device))
    }
}

#[cfg(test)]
mod tests {
    use bake_macros::Batchable;
use burn::{Tensor, tensor::{Device, Int, Shape}};
    use crate::data::batchable::Batchable;

    #[test]
    fn cat_test() {
        let device = Device::default();
        let t1 = Tensor::<3>::from_floats([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]], &device);
        let t2 = Tensor::<3>::from_floats([[[2.0, 3.0, 4.0, 5.0], [3.0, 2.0, 9.0, 10.0]]], &device);
        let catanated_answer = Tensor::<3>::from_floats([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], [[2.0, 3.0, 4.0, 5.0], [3.0, 2.0, 9.0, 10.0]]], &device);
        let catanated = Batchable::cat(vec![t1, t2]);
        let a = catanated_answer.equal(catanated).all().into_scalar::<bool>();
        assert!(a);
    }

    #[test]
    fn select_test() {
        let device = Device::default();
        let c = Tensor::<1>::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &device);
        let idx = Tensor::<1, Int>::from_ints([1, 5, 7], &device);
        let selected = Batchable::select(c, idx);
        let selected_answer = Tensor::<1>::from_floats([1.0, 5.0, 7.0], &device);
        assert!(selected.equal(selected_answer).all().into_scalar::<bool>());
    }

    #[test]
    fn slice_test() {
        let device = Device::default();
        let c = Tensor::<1>::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &device);
        let sliced = Batchable::slice(c, 2..5);
        let sliced_answer = Tensor::<1>::from_floats([2.0, 3.0, 4.0], &device);
        assert!(sliced.equal(sliced_answer).all().into_scalar::<bool>());
    }

    #[test]
    fn zero_like_test() {
        #[derive(Batchable, Clone, Debug)]
        struct TestStruct {
            pub t1: Tensor<1>,
            pub t2: Tensor<2>,
        }
        let device = Device::default();
        let t1 = Tensor::<1>::from_floats([0.0], &device);
        let t2 = Tensor::<2>::from_floats([[3.0, 4.0]], &device);
        let st = TestStruct { t1, t2 };

        let st = TestStruct::zeros_like(10, &st, &device);
        assert!(st.t1.shape() == Shape::new([10]));
        assert!(st.t2.shape() == Shape::new([10, 2]));
    }
}