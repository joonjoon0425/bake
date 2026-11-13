//! one-hot space transformer
use crate::core::env::space::{Space, BatchedSpace, DiscreteSpace, BatchedDiscreteSpace, ContinuousSpace, BatchedContinuousSpace};
use crate::core::env::space_transform::{BatchedSpaceTransform, SpaceTransform};

use ndarray::prelude::*;
use burn::prelude::*;

/// one-hot transformer
pub struct OneHotTransform;
impl OneHotTransform { 
    /// create new `OneHotTransform`
    pub fn new() -> OneHotTransform { OneHotTransform }
}

impl SpaceTransform<DiscreteSpace> for OneHotTransform {
    type Applied = ContinuousSpace<1>;

    fn transform_space(&self, applicant: &DiscreteSpace) -> Self::Applied {
        let n_obs = applicant.n();
        let low = ArrayD::from_shape_vec(vec![n_obs], vec![0.; n_obs]).unwrap();
        let high = ArrayD::from_shape_vec(vec![n_obs], vec![1.; n_obs]).unwrap();
        ContinuousSpace::new(&[n_obs], low, high)
    }

    fn transform_element(&mut self, applicant: &DiscreteSpace, element: <DiscreteSpace as Space>::Element) -> <Self::Applied as Space>::Element {
        let n_obs = applicant.n();
        let mut vec = vec![0.; n_obs]; vec[element] = 1.;
        ArrayD::from_shape_vec(vec![n_obs], vec).unwrap()
    }
}

/// one-hot transformer
pub struct BatchedOneHotTransform;
impl BatchedOneHotTransform { 
    /// create new `BatchedOneHotTransform`
    pub fn new() -> BatchedOneHotTransform { BatchedOneHotTransform }
}

impl BatchedSpaceTransform<BatchedDiscreteSpace> for BatchedOneHotTransform {
    type Applied = BatchedContinuousSpace<2>;

    fn transform_space(&self, applicant: &BatchedDiscreteSpace) -> Self::Applied {
        let n_obs = applicant.n();
        let device = applicant.device();
        let low = Tensor::from_data(TensorData::new(vec![0; n_obs], [1, n_obs]), &device);
        let high = Tensor::from_data(TensorData::new(vec![1; n_obs], [1, n_obs]), &device);
        BatchedContinuousSpace::new(&[1, n_obs], low, high, &device)
    }

    fn transform_element(&mut self, applicant: &BatchedDiscreteSpace, element: <BatchedDiscreteSpace as BatchedSpace>::Element) -> <Self::Applied as BatchedSpace>::Element {
        element.one_hot(applicant.n()).float()
    }
}