//! discrete constraint

use burn::prelude::*;
use ndarray::Array2;
use crate::core::{Vector, Repr, Single, constraint::Unconstrained};

/// discrete constraint for discrete action spaces
pub trait DiscreteConstraint<M: Repr> {
    /// Fill the given `values` with `fill_value` where mask is false
    fn apply(self, values: M::FloatArray<2>, fill_value: f32) -> M::FloatArray<2>;
    /// Compute the mean, considering the constraints. squeezed after reduction
    fn mean_dim(self, values: M::FloatArray<2>) -> M::Float;
    /// return the possible number of actions
    fn n_possible_actions(&self) -> Option<M::Int>;
    /// returns if the given action is possible
    fn is_possible(&self, action: M::Int) -> M::Bool;
}

/// implementation for tensor mask
impl DiscreteConstraint<Vector> for Tensor<2, Bool> {
    fn apply(self, values: Tensor<2>, fill_value: f32) -> Tensor<2> {
        values.mask_fill(self.bool_not(), fill_value)
    }

    fn mean_dim(self, values: Tensor<2>) -> Tensor<1> {
        let invalid = self.clone().bool_not();
        let n_possible_actions = self.float().sum_dim(1);
        let mean = values.mask_fill(invalid, 0f32).sum_dim(1) / n_possible_actions;
        mean.squeeze_dim(1)
    }

    fn n_possible_actions(&self) -> Option<Tensor<1, Int>> {
        Some(self.clone().float().sum_dim(1).squeeze_dim(1).int())
    }

    fn is_possible(&self, action: Tensor<1, Int>) -> Tensor<1, Bool> {
        self.clone().gather(1, action.unsqueeze_dim(1)).squeeze_dim(1)
    }
}

/// implementation for `Unconstrained`
impl DiscreteConstraint<Vector> for Unconstrained {
    fn apply(self, values: Tensor<2>, _: f32) -> Tensor<2> { values }
    fn mean_dim(self, values: Tensor<2>) -> Tensor<1> { values.mean_dim(1).squeeze_dim(1) }
    fn n_possible_actions(&self) -> Option<Tensor<1, Int>> { None }
    fn is_possible(&self, action: Tensor<1, Int>) -> Tensor<1, Bool> {
        let n_actions = action.shape()[0];
        Tensor::from_bool(vec![true; n_actions].as_slice(), &action.device())
    }
}

impl DiscreteConstraint<Single> for Unconstrained {
    fn apply(self, values: <Single as Repr>::FloatArray<2>, _: f32) -> <Single as Repr>::FloatArray<2> { values }
    fn mean_dim(self, values: <Single as Repr>::FloatArray<2>) -> <Single as Repr>::Float { values.mean_axis(ndarray::Axis(1)).unwrap().to_vec()[0] }
    fn n_possible_actions(&self) -> Option<<Single as Repr>::Int> { None }
    fn is_possible(&self, _: <Single as Repr>::Int) -> <Single as Repr>::Bool { true }
}

#[cfg(test)]
mod tests {
    use crate::core::constraint::discrete::DiscreteConstraint;
    use burn::prelude::*;

    #[test]
    fn apply_test() {
        let device = Device::flex();
        let t = Tensor::<2>::from_floats([[1f32, 2., 3., 4.], [1f32, 2., 3., 4.], [1f32, 2., 3., 4.]], &device);
        let m = Tensor::<2, Bool>::from_bool([[true, true, false, true], [false, true, true, true], [true, true, false, false]], &device);

        let t = m.apply(t, -1e9);
        let ans = Tensor::<2>::from_floats([[1f32, 2., -1e9, 4.], [-1e9, 2., 3., 4.], [1f32, 2., -1e9, -1e9]], &device);
        assert!(t.equal(ans).all().into_scalar::<bool>())
    }

    #[test]
    fn mean_dim_test() {
        let device = Device::default();
        let mask = Tensor::from_bool([[false, true, true, true], [true, true, false, false], [true, true, false, true]], &device);
        let values = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [10.0, 20.0, 30.0, 40.0]], &device);

        let mean: Vec<f32> = mask.mean_dim(values).into_data().try_into_vec().unwrap();
        assert!(mean[0] == 3.0);
        assert!(mean[1] == -1.5);
        assert!(mean[2] == 70.0 / 3.0);
    }

    #[test]
    fn n_possible_actions_test() {
        let device = Device::default();
        let mask = Tensor::from_bool([[false, true, true, true], [true, true, false, false], [true, true, false, true]], &device);
        let n = mask.n_possible_actions().unwrap();
        let ans = Tensor::<1, Int>::from_ints([3, 2, 3], &device);

        assert!(ans.equal(n).all().into_scalar::<bool>())
    }

    #[test]
    fn possible_actions_test() {
        let device = Device::default();
        let mask = Tensor::from_bool([[false, true, true, true], [true, true, false, false], [true, true, false, true]], &device);
        let action_1 = Tensor::<1, Int>::from_ints([1, 0, 2], &device);
        let b = mask.is_possible(action_1);
        let ans = Tensor::from_bool([true, true, false], &device);
        assert!(b.equal(ans).all().into_scalar::<bool>())
    }
}