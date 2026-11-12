//! Byte (u8) space
use super::{BatchedSpace, Space, IntoBatchedSpace};
use burn::prelude::*;
use rand::prelude::*;
use ndarray::prelude::*;

/// Byte space with feature rank D
#[derive(Debug, Clone)]
pub struct ByteSpace<const D: usize> {
    shape: [usize; D],
}

impl<const D: usize> ByteSpace<D> {
    /// create a new byte space, with feature rank D
    pub fn new(shape: &[usize]) -> Self {
        Self { shape: shape.try_into().unwrap() }
    }
    /// return the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<const D: usize> Space for ByteSpace<D> {
    type Element = ArrayD<u8>;

    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Element {
        let n_elems = self.shape.iter().product();
        let elems: Vec<u8> = (0..n_elems).into_iter().map(|_| rng.random_range(0u8..255u8)).collect();
        let s = ArrayD::from_shape_vec(self.shape(), elems).unwrap();
        s
    }
}

/// Byte space with feature rank D
#[derive(Debug, Clone)]
pub struct BatchedByteSpace<const D: usize> {
    shape: [usize; D],
    device: Device,
}

impl<const D: usize> BatchedByteSpace<D> {
    /// create a new byte space, with feature rank D
    pub fn new(shape: &[usize], device: &Device) -> Self {
        Self { shape: shape.try_into().unwrap(), device: device.clone() }
    }
    /// return the shape
    /// # Warning
    /// - Due to fixed rank of `burn::Tensor`, shape will always have dimension 1 in the first dimension
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<const D: usize> BatchedSpace for BatchedByteSpace<D> {
    type Element = Tensor<D, Int>;

    fn sample(&self) -> Self::Element {
        let shape = self.shape();
        Tensor::random(shape, burn::tensor::Distribution::Uniform(0., 255.), (&self.device, burn::tensor::DType::U8))
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}

macro_rules! impl_to_batched_byte {
    ($d:literal) => {
        
        impl IntoBatchedSpace for ByteSpace<$d> {
            type Batched = BatchedByteSpace<{$d + 1}>;

            fn into_batched(spaces: &[Self], device: &Device) -> Self::Batched {
                assert!(!spaces.is_empty(), "IntoBatchedSpace::into_batched: got empty space vector");
                let single_shape = spaces[0].shape();
                assert!(spaces.iter().all(|s| s.shape() == single_shape), "IntoBatchedSpace::into_batched: the shape of each `ByteSpace` is not same");
                let mut shape = vec![spaces.len()];
                shape.extend_from_slice(single_shape);
                Self::Batched::new(&shape, device)
            }

            fn bind(elems: Vec<Self::Element>, device: &Device) -> <Self::Batched as BatchedSpace>::Element {
                assert!(!elems.is_empty(), "ToBatchedSpace::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
                let n = elems.len();
                let mut shape = elems[0].shape().to_vec(); shape.insert(0, n);
                let native = elems.into_iter().flatten().collect();
                Tensor::from_data(TensorData::new(native, shape), (device, burn::tensor::DType::U8))
            }

            fn unbind(elem: <Self::Batched as BatchedSpace>::Element) -> Vec<Self::Element> {
                let shape = elem.dims();
                let n_elem = shape.iter().skip(1).product();
                let flattened = elem.into_data().try_into_vec_as::<u8>().unwrap();
                
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

impl_to_batched_byte!(1);
impl_to_batched_byte!(2);
impl_to_batched_byte!(3);
impl_to_batched_byte!(4);
impl_to_batched_byte!(5);
impl_to_batched_byte!(6);
impl_to_batched_byte!(7);
impl_to_batched_byte!(8);
impl_to_batched_byte!(9);