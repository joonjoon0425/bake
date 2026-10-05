//! Action mask trait and implementations for masking
use std::fmt::Debug;
use burn::{Tensor, tensor::Bool};
use crate::{constraint::Unconstrained, data::batchable::Batchable};

/// Discrete constraint trait for discrete action types
pub trait DiscreteConstraint<const D: usize = 2> : Debug + Clone + Batchable {
    /// Fill the given `values` with `fill_value` where mask is true
    fn apply(self, values: Tensor<D>, fill_value: f32) -> Tensor<D>;
    /// Compute the mean, considering the constraints.
    fn mean_dim(self, dim: usize, values: Tensor<D>) -> Tensor<D>;
}

/// Basic Discrete action mask for tensors
#[derive(Debug, Clone)]
pub struct DiscreteMask<const D: usize = 2>(pub Tensor<D, Bool>);
impl<const D: usize> DiscreteConstraint<D> for DiscreteMask<D> {
    fn apply(self, values: Tensor<D>, fill_value: f32) -> Tensor<D> {
        values.mask_fill(self.0.bool_not(), fill_value)
    }

    fn mean_dim(self, dim: usize, values: Tensor<D>) -> Tensor<D> {
        let invalid = self.clone().0.bool_not();
        let n_possible_actions = self.0.float().sum_dim(dim);
        let mean = values.mask_fill(invalid.clone(), 0f32).sum_dim(dim) / n_possible_actions;
        mean
    }
}

/// `DiscreteConstraint` implementation for `Unconstrained`
impl<const D: usize> DiscreteConstraint<D> for Unconstrained {
    fn apply(self, values: Tensor<D>, _: f32) -> Tensor<D> { values }
    fn mean_dim(self, dim: usize, values: Tensor<D>) -> Tensor<D> { values.mean_dim(dim) }
}

#[cfg(test)]
mod tests {
    use burn::{Tensor, tensor::Device};
    use crate::constraint::discrete_constraint::{DiscreteConstraint, DiscreteMask};

    #[test]
    fn apply_test() {
        let device = Device::default();
        let mask = DiscreteMask::<2>(Tensor::from_bool([[false, true, true, true]], &device));
        let values = Tensor::<2>::from_floats([[1.1, 1.2, -1.4, 5.6]], &device);

        let masked_values = mask.clone().apply(values.clone(), -1e9);
        let mask_from = values.equal(masked_values);

        assert!(mask.0.equal(mask_from).all().into_scalar::<bool>(), "mask and mask_from must equal");
    }

    #[test]
    fn mean_dim_test() {
        let device = Device::default();
        let mask = DiscreteMask::<2>(Tensor::from_bool([[false, true, true, true], [true, true, false, false], [true, true, false, true]], &device));
        let values = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [10.0, 20.0, 30.0, 40.0]], &device);

        let mean: Vec<f32> = mask.mean_dim(0, values).into_data().try_into_vec().unwrap();
        assert!(mean[0] == 4.5);
        assert!(mean[1] == 20.0 / 3.0);
        assert!(mean[2] == 3.0);
        assert!(mean[3] == 22.0);
    }
}