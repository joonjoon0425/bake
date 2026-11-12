//! Space for continuous elements
use super::{Space, BatchedSpace, IntoBatchedSpace};
use burn::{prelude::*, tensor::Distribution::Uniform};
use rand::prelude::*;
use ndarray::prelude::*;

/// Continuous space with rank `D`
#[derive(Debug, Clone)]
pub struct ContinuousSpace<const D: usize> {
    shape: [usize; D],
    low: ArrayD<f32>,
    high: ArrayD<f32>,
}

impl<const D: usize> ContinuousSpace<D> {
    /// create a new continuous space, with feature rank D
    pub fn new(shape: &[usize], low: ArrayD<f32>, high: ArrayD<f32>) -> Self {
        assert!(shape == low.shape(), "`low` and given shape mismatch: low: {:?}, shape: {:?}", low, shape);
        assert!(shape == high.shape(), "`high` and given shape mismatch");
        assert!(low.iter().zip(high.iter()).all(|(l, h)| l <= h), "Some elements of `low` is higher than `high`");
        Self { shape: shape.try_into().unwrap(), low, high }
    }
    /// return the shape
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

impl<const D: usize> Space for ContinuousSpace<D> {
    type Element = ArrayD<f32>;

    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Element {
        let n_elems = self.shape.iter().product();
        let elems: Vec<f32> = (0..n_elems).into_iter().map(|_| rng.random_range(0f32..1f32)).collect();
        let s = ArrayD::from_shape_vec(self.shape(), elems).unwrap();
        (self.high.clone() - self.low.clone()) * s + self.low.clone()
    }
}

/// Continuous space with rank `D`
#[derive(Debug, Clone)]
pub struct BatchedContinuousSpace<const D: usize> {
    shape: [usize; D],
    low: Tensor<D>,
    high: Tensor<D>,
    device: Device,
}

impl<const D: usize> BatchedContinuousSpace<D> {
    /// create a new continuous space, with feature rank `D`
    pub fn new(shape: &[usize], low: Tensor<D>, high: Tensor<D>, device: &Device) -> Self {
        assert!(shape == low.dims(), "`low` and given shape mismatch: low: {:?}, shape: {:?}", low.dims(), shape);
        assert!(shape == high.dims(), "`high` and given shape mismatch");
        assert!(low.dtype() == high.dtype(), "`low` and `high` dtype mismatch");
        assert!(low.clone().lower_equal(high.clone()).all().into_scalar::<bool>(), "Some elements of `low` is higher than `high`");
        Self { shape: shape.try_into().unwrap(), low, high, device: device.clone() }
    }
    /// return the shape
    /// # Warning
    /// - note that shape contains batch dimension in frontmost axis
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

impl<const D: usize> BatchedSpace for BatchedContinuousSpace<D> {
    type Element = Tensor<D>;

    fn sample(&self) -> Self::Element {
        let shape = self.shape.clone();
        let s = Tensor::random(shape, Uniform(0., 1.), &self.device);
        (self.high.clone() - self.low.clone()) * s + self.low.clone()
    }
    
    fn device(&self) -> Device {
        self.device.clone()
    }
}

macro_rules! impl_to_batched_continuous {
    ($d:literal) => {
        impl IntoBatchedSpace for ContinuousSpace<$d> {
            type Batched = BatchedContinuousSpace<{$d + 1}>;

            fn into_batched(spaces: &[Self], device: &Device) -> Self::Batched {
                assert!(!spaces.is_empty(), "IntoBatchedSpace::into_batched: got empty space vector");
                let single_shape = spaces[0].shape();
                assert!(spaces.iter().all(|s| s.shape() == single_shape), "IntoBatchedSpace::into_batched: the shape of each `ContinuousSpace` is not same");
                let mut shape = vec![spaces.len()];
                shape.extend_from_slice(single_shape);

                let low = spaces.iter().map(|s| s.low.iter()).flatten().copied().collect();
                let low = Tensor::from_data(TensorData::new(low, &shape), device);

                let high = spaces.iter().map(|s| s.low.iter()).flatten().copied().collect();
                let high = Tensor::from_data(TensorData::new(high, &shape), device);

                Self::Batched::new(&shape, low, high, device)
            }

            fn bind(elems: Vec<Self::Element>, device: &Device) -> <Self::Batched as BatchedSpace>::Element {
                assert!(!elems.is_empty(), "ToBatchedSpace::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
                let n = elems.len();
                let mut shape = elems[0].shape().to_vec(); shape.insert(0, n);
                let native = elems.into_iter().flatten().collect();
                Tensor::from_data(TensorData::new(native, shape), device)
            }

            fn unbind(elem: <Self::Batched as BatchedSpace>::Element) -> Vec<Self::Element> {
                let shape = elem.dims();
                let n_elem = shape.iter().skip(1).product();
                let flattened = elem.into_data().try_into_vec_as::<f32>().unwrap();
                
                let chunks = flattened.chunks_exact(n_elem);
                let mut vec = Vec::with_capacity(shape[0]);

                for chunk in chunks {
                    vec.push(ArrayD::from_shape_vec(shape[1..].to_vec(), chunk.to_vec()).unwrap())
                }
                vec
            }
        }        
    };
}

impl_to_batched_continuous!(1);
impl_to_batched_continuous!(2);
impl_to_batched_continuous!(3);
impl_to_batched_continuous!(4);
impl_to_batched_continuous!(5);
impl_to_batched_continuous!(6);
impl_to_batched_continuous!(7);
impl_to_batched_continuous!(8);
impl_to_batched_continuous!(9);