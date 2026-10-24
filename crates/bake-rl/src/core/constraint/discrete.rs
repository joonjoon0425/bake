//! discrete constraint

use burn::prelude::*;
use ndarray::prelude::*;

use crate::core::data::{BatchEmpty, BatchScalar};
use crate::core::data::batch_dim::convert::DetachBatchDim;
use crate::core::{Repr, Single, Vector};
use crate::core::constraint::Unconstrained;
use crate::core::data::batch_dim::{BatchTensor, convert::AttachBatchDim};

/// discrete constraint for discrete action spaces
pub trait DiscreteConstraint<M: Repr> {
    /// Fill the given `values` with `fill_value` where mask is false
    fn apply(self, values: <M::Array<2, Float> as AttachBatchDim>::Attached, fill_value: f32) -> <M::Array<2, Float> as AttachBatchDim>::Attached;
    /// Compute the mean, considering the constraints. squeezed after reduction
    fn mean_dim(self, values: <M::Array<2, Float> as AttachBatchDim>::Attached) -> <M::Array<1, Float> as AttachBatchDim>::Attached;
    /// return the possible number of actions
    fn n_possible_actions(&self) -> Option<M::Scalar<Int>>;
    /// returns if the given action is possible
    fn is_possible(&self, action: <M::Scalar<Int> as AttachBatchDim>::Attached) -> <M::Scalar<Bool> as AttachBatchDim>::Attached;
}

/// implementation for tensor mask
impl DiscreteConstraint<Vector> for BatchTensor<2, Bool> {
    fn apply(self, values: BatchTensor<2>, fill_value: f32) -> BatchTensor<2> {
        let (mask, bs_mask) = self.detach_batch_dim();
        let (values, bs_value) = values.detach_batch_dim();
        assert!(bs_mask == bs_value, "batch dimension does not matches");

        values.mask_fill(mask.bool_not(), fill_value).attach_batch_dim(bs_value)
    }

    fn mean_dim(self, values: BatchTensor<2>) -> BatchTensor<1> {
        let (mask, bs_mask) = self.detach_batch_dim();
        let (values, bs_value) = values.detach_batch_dim();
        assert!(bs_mask == bs_value, "batch dimension does not matches");

        let invalid = mask.clone().bool_not();
        let n_possible_actions = mask.float().sum_dim(1);
        let mean = values.mask_fill(invalid, 0f32).sum_dim(1) / n_possible_actions;
        mean.squeeze_dim(1).attach_batch_dim(bs_value)
    }

    fn n_possible_actions(&self) -> Option<Tensor<1, Int>> {
        let (mask, _) = self.clone().detach_batch_dim();
        Some(mask.float().sum_dim(1).squeeze_dim(1).int())
    }

    fn is_possible(&self, action: BatchTensor<1, Int>) -> BatchTensor<1, Bool> {
        let (mask, bs_mask) = self.clone().detach_batch_dim();
        let (action, bs_value) = action.detach_batch_dim();
        assert!(bs_mask == bs_value, "batch dimension does not matches");
        mask.gather(1, action.unsqueeze_dim(1)).squeeze_dim(1).attach_batch_dim(bs_mask)
    }
}

/// implementation for `Unconstrained`
impl DiscreteConstraint<Vector> for BatchEmpty<Unconstrained> {
    fn apply(self, values: <<Vector as Repr>::Array<2, Float> as AttachBatchDim>::Attached, _: f32) -> <<Vector as Repr>::Array<2, Float> as AttachBatchDim>::Attached {
        values
    }

    fn mean_dim(self, values: <<Vector as Repr>::Array<2, Float> as AttachBatchDim>::Attached) -> <<Vector as Repr>::Array<1, Float> as AttachBatchDim>::Attached {
        let (t, batch_shape) = values.detach_batch_dim();
        let t = t.mean_dim(1).squeeze_dim(1);
        t.attach_batch_dim(batch_shape)
    }

    fn is_possible(&self, action: <<Vector as Repr>::Scalar<Int> as AttachBatchDim>::Attached) -> <<Vector as Repr>::Scalar<Bool> as AttachBatchDim>::Attached {
        let (t, bs) = action.detach_batch_dim();
        let bt = Tensor::from_bool(vec![true; t.shape().num_elements()].as_slice(), &t.device());
        bt.attach_batch_dim(bs)
    }

    fn n_possible_actions(&self) -> Option<<Vector as Repr>::Scalar<Int>> {
        None
    }
}

impl DiscreteConstraint<Single> for BatchEmpty<Unconstrained> {
    fn apply(self, values: <<Single as Repr>::Array<2, Float> as AttachBatchDim>::Attached, _: f32) -> <<Single as Repr>::Array<2, Float> as AttachBatchDim>::Attached {
        values
    }

    fn mean_dim(self, values: <<Single as Repr>::Array<2, Float> as AttachBatchDim>::Attached) -> <<Single as Repr>::Array<1, Float> as AttachBatchDim>::Attached {
        let (t, bs) = values.detach_batch_dim();
        let n = t.ndim() - 1;
        t.mean_axis(Axis(n)).unwrap().attach_batch_dim(bs)
    }

    fn is_possible(&self, action: <<Single as Repr>::Scalar<Int> as AttachBatchDim>::Attached) -> <<Single as Repr>::Scalar<Bool> as AttachBatchDim>::Attached {
        let (_, bs) = action.detach_batch_dim();
        let b = ArrayD::from_elem(bs.clone(), true);
        BatchScalar::Many(b.attach_batch_dim(bs))
    }

    fn n_possible_actions(&self) -> Option<<Single as Repr>::Scalar<Int>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{constraint::discrete::DiscreteConstraint, data::batch_dim::convert::{AttachBatchDim, DetachBatchDim}};
    use burn::prelude::*;

    #[test]
    fn apply_test() {
        let device = Device::flex();
        let t = Tensor::<2>::from_floats([[1f32, 2., 3., 4.], [1f32, 2., 3., 4.], [1f32, 2., 3., 4.]], &device).attach_batch_dim(vec![1]);
        let m = Tensor::<2, Bool>::from_bool([[true, true, false, true], [false, true, true, true], [true, true, false, false]], &device).attach_batch_dim(vec![1]);

        let (t, _) = m.apply(t, -1e9).detach_batch_dim();
        let ans = Tensor::<2>::from_floats([[1f32, 2., -1e9, 4.], [-1e9, 2., 3., 4.], [1f32, 2., -1e9, -1e9]], &device);
        assert!(t.equal(ans).all().into_scalar::<bool>())
    }

    #[test]
    fn mean_dim_test() {
        let device = Device::default();
        let mask = Tensor::from_bool([[false, true, true, true], [true, true, false, false], [true, true, false, true]], &device).attach_batch_dim(vec![1]);
        let values = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [10.0, 20.0, 30.0, 40.0]], &device).attach_batch_dim(vec![1]);

        let mean: Vec<f32> = mask.mean_dim(values).detach_batch_dim().0.into_data().try_into_vec().unwrap();
        assert!(mean[0] == 3.0);
        assert!(mean[1] == -1.5);
        assert!(mean[2] == 70.0 / 3.0);
    }

    #[test]
    fn n_possible_actions_test() {
        let device = Device::default();
        let mask = Tensor::from_bool([[false, true, true, true], [true, true, false, false], [true, true, false, true]], &device).attach_batch_dim(vec![1]);
        let n = mask.n_possible_actions().unwrap();
        let ans = Tensor::<1, Int>::from_ints([3, 2, 3], &device);

        assert!(ans.equal(n).all().into_scalar::<bool>())
    }

    #[test]
    fn possible_actions_test() {
        let device = Device::default();
        let mask = Tensor::from_bool([[false, true, true, true], [true, true, false, false], [true, true, false, true]], &device).attach_batch_dim(vec![1]);
        let action_1 = Tensor::<1, Int>::from_ints([1, 0, 2], &device).attach_batch_dim(vec![1]);
        let b = mask.is_possible(action_1);
        let ans = Tensor::from_bool([true, true, false], &device);
        assert!(b.detach_batch_dim().0.equal(ans).all().into_scalar::<bool>())
    }
}