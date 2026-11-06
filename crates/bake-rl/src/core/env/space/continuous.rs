//! Space for continuous elements
use super::{NativeSpace, BurnSpace};
use burn::{prelude::*, tensor::Distribution::Uniform};
use rand::prelude::*;
use ndarray::prelude::*;

/// Continuous space with feature rank (D - 1)
/// - E: element type
///     - `burn::Tensor`
#[derive(Debug, Clone)]
pub struct ContinuousSpace<Repr: ContinuousSpaceRepr, const D: usize> {
    shape: [usize; D],
    low: Repr::Bound<D>,
    high: Repr::Bound<D>,
    inner: Repr::Inner,
}

/// Continuous Space Representation
pub trait ContinuousSpaceRepr {
    type Bound<const D: usize>;
    type Inner;
}

impl ContinuousSpaceRepr for crate::core::Native { type Bound<const D: usize> = ArrayD<f32>; type Inner = SmallRng; }
impl ContinuousSpaceRepr for crate::core::Burn { type Bound<const D: usize> = Tensor<D>; type Inner = Device; }

impl<const D: usize> ContinuousSpace<crate::core::Burn, D> {
    /// create a new continuous space, with feature rank (D - 1) as `Burn` representation
    /// # Warning
    /// - note that D is (feature rank + 1)
    /// - this is due to the fixed rank of `burn::Tensor`
    /// - Even if it is in `Native` representation, for unity
    pub fn burn(shape: &[usize], low: Tensor<D>, high: Tensor<D>, device: &Device) -> Self {
        assert!(shape.iter().zip(low.dims()).all(|(s, l)| *s == l), "`low` and given shape mismatch");
        assert!(shape.iter().zip(high.dims()).all(|(s, l)| *s == l), "`high` and given shape mismatch");
        assert!(low.dtype() == high.dtype(), "`low` and `high` dtype mismatch");
        assert!(low.clone().lower_equal(high.clone()).all().into_scalar::<bool>(), "Some elements of `low` is higher than `high`");
        Self { shape: shape.try_into().unwrap(), low, high, inner: device.clone() }
    }
    /// return the shape
    /// # Warning
    /// - Due to fixed rank of `burn::Tensor`, shape will always have dimension 1 in the first dimension
    /// - Even if it is in `Native` representation
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
}

impl<const D: usize> BurnSpace for ContinuousSpace<crate::core::Burn, D> {
    type Element = Tensor<D>;

    fn sample(&self, n: usize) -> Self::Element {
        let mut shape = self.shape.clone();
        shape[0] = n;
        let s = Tensor::random(shape, Uniform(0., 1.), &self.inner);
        (self.high.clone() - self.low.clone()) * s + self.low.clone()
    }
    
    fn device(&self) -> Device {
        self.inner.clone()
    }
}

impl<const D: usize> ContinuousSpace<crate::core::Native, D> {
    /// create a new continuous space, with feature rank (D - 1) as `Native` representation
    /// # Warning
    /// - note that D is (feature rank + 1)
    /// - this is due to the fixed rank of `burn::Tensor`
    /// - Even if it is in `Native` representation, for unity
    pub fn native(shape: &[usize], low: ArrayD<f32>, high: ArrayD<f32>, seed: u64) -> Self {
        assert!(shape.iter().zip(low.shape()).all(|(s, l)| s == l), "`low` and given shape mismatch");
        assert!(shape.iter().zip(high.shape()).all(|(s, l)| s == l), "`high` and given shape mismatch");
        assert!(low.iter().zip(high.iter()).all(|(l, h)| l <= h), "Some elements of `low` is higher than `high`");
        Self { shape: shape.try_into().unwrap(), low, high, inner: SmallRng::seed_from_u64(seed) }
    }
    /// return the shape
    /// # Warning
    /// - Due to fixed rank of `burn::Tensor`, shape will always have dimension 1 in the first dimension
    /// - Even if it is in `Native` representation
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// return the lower bound
    pub fn low(&self) -> &ArrayD<f32> {
        &self.low
    }
    /// return the upper bound
    pub fn high(&self) -> &ArrayD<f32> {
        &self.high
    }
}


impl<const D: usize> NativeSpace for ContinuousSpace<crate::core::Native, D> {
    type Element = ArrayD<f32>;

    fn sample(&mut self) -> Self::Element {
        let n_elems = self.shape.iter().product();
        let elems: Vec<f32> = (0..n_elems).into_iter().map(|_| self.inner.random_range(0f32..1f32)).collect();
        let s = ArrayD::from_shape_vec(self.shape(), elems).unwrap();
        (self.high.clone() - self.low.clone()) * s + self.low.clone()
    }
}