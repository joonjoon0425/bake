//! A Batchable trait

use core::ops::Range;

use burn::{
    Tensor,
    tensor::{Bool, Int},
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
}

#[cfg(test)]
mod tests {
    use burn::{Tensor, tensor::{Device, Int}};
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
}