//! Space for continuous elements
use super::Space;
use burn::{prelude::*, tensor::Distribution::Uniform};
/// Continuous space with feature rank (D - 1)
/// - E: element type
///     - `burn::Tensor`
#[derive(Debug, Clone)]
pub struct ContinuousSpace<const D: usize> {
    device: Device,
    shape: Vec<usize>,
    low: Tensor<D>,
    high: Tensor<D>,
    dtype: burn::tensor::DType,
}

impl<const D: usize> ContinuousSpace<D> {
    /// create a new continuous space, with feature rank (D - 1) as `Burn` representation
    /// # Warning
    /// - note that D is (feature rank + 1)
    /// - this is due to the fixed rank of `burn::Tensor`
    pub fn burn(shape: &[usize], low: Tensor<D>, high: Tensor<D>, device: &Device) -> Self {
        assert!(shape.iter().zip(low.dims()).all(|(s, l)| *s == l), "`low` and given shape mismatch");
        assert!(shape.iter().zip(high.dims()).all(|(s, l)| *s == l), "`high` and given shape mismatch");
        assert!(low.dtype() == high.dtype(), "`low` and `high` dtype mismatch");
        assert!(low.clone().lower_equal(high.clone()).all().into_scalar::<bool>(), "Some elements of `low` is higher than `high`");
        let dtype = low.dtype();
        Self { shape: shape.try_into().unwrap(), low, high, device: device.clone(), dtype }
    }
    /// return the shape
    /// # Warning
    /// - Due to fixed rank of `burn::Tensor`, shape will always have dimension 1 in the first dimension
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// return the lower bound
    pub fn low(&self) -> &Tensor<D> {
        &self.low
    }
    /// return the upper bound
    pub fn high(&self) -> &Tensor<D> {
        &self.high
    }
    /// return the device
    pub fn device(&self) -> Device {
        self.device.clone()
    }
}

impl<const D: usize> Space for ContinuousSpace<D> {
    type Element = Tensor<D>;

    fn sample(&mut self) -> Self::Element {
        let s = Tensor::random(self.shape(), Uniform(0., 1.), (&self.device, self.dtype));
        (self.high.clone() - self.low.clone()) * s + self.low.clone()
    }
}