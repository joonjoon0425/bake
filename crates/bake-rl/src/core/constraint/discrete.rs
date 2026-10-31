//! discrete constraint

use burn::prelude::*;

use crate::core::data::BatchOps;
/// discrete constraint for discrete action spaces
pub trait DiscreteConstraint: BatchOps {
    /// Fill the given `values` with `fill_value` where mask is false
    fn apply(self, values: Tensor<2>, fill_value: f32) -> Tensor<2>;
    /// Compute the mean, considering the constraints. squeezed after reduction
    fn mean_dim(self, values: Tensor<2>) -> Tensor<1>;
    /// return the possible number of actions
    fn n_possible_actions(&self) -> Option<Tensor<1, Int>>;
    /// returns if the given action is possible
    fn is_possible(&self, action: Tensor<1, Int>) -> Tensor<1, Bool>;
}

/// implementation for tensor mask
impl DiscreteConstraint for Tensor<2, Bool> {
    fn apply(self, values: Tensor<2>, fill_value: f32) -> Tensor<2> {
        assert!(self.dims()[0] == values.dims()[0], "DiscreteConstraint::apply: batch dimension mismatch");
        values.mask_fill(self.bool_not(), fill_value)
    }

    fn mean_dim(self, values: Tensor<2>) -> Tensor<1> {
        assert!(self.dims()[0] == values.dims()[0], "DiscreteConstraint::mean_dim: batch dimension mismatch");

        let invalid = self.clone().bool_not();
        let n_possible_actions = self.float().sum_dim(1);
        let mean = values.mask_fill(invalid, 0f32).sum_dim(1) / n_possible_actions;
        mean.squeeze_dim(1)
    }

    fn n_possible_actions(&self) -> Option<Tensor<1, Int>> {
        Some(self.clone().float().sum_dim(1).squeeze_dim(1).int())
    }

    fn is_possible(&self, action: Tensor<1, Int>) -> Tensor<1, Bool> {
        assert!(self.dims()[0] == action.dims()[0], "DiscreteConstraint::is_possible: batch dimension does not mismatch");
        self.clone().gather(1, action.unsqueeze_dim(1)).squeeze_dim(1)
    }
}

