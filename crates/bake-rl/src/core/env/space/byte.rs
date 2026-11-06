//! Byte (u8) space
use super::{BurnSpace, NativeSpace, ToBurnSpace};
use burn::prelude::*;
use rand::prelude::*;
use ndarray::prelude::*;

/// Byte space with feature rank (D - 1)
#[derive(Debug, Clone)]
pub struct ByteSpace<Repr: ByteSpaceRepr, const D: usize> {
    shape: [usize; D],
    inner: Repr::Inner,
}

/// Byte Space Representation
pub trait ByteSpaceRepr { 
    /// inner types which each representation requires
    type Inner;
}
impl ByteSpaceRepr for crate::core::Native { type Inner = SmallRng; }
impl ByteSpaceRepr for crate::core::Burn { type Inner = Device; }

impl<const D: usize> ByteSpace<crate::core::Burn, D> {
    /// create a new byte space, with feature rank (D - 1) with `Burn` representation
    /// # Warning
    /// - note that D is (feature rank + 1)
    /// - this is due to the fixed rank of `burn::Tensor`
    /// - Even if it is in `Native` representation, for unity
    pub fn burn(shape: &[usize], device: &Device) -> Self {
        Self { shape: shape.try_into().unwrap(), inner: device.clone() }
    }
    /// return the shape
    /// # Warning
    /// - Due to fixed rank of `burn::Tensor`, shape will always have dimension 1 in the first dimension
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<const D: usize> BurnSpace for ByteSpace<crate::core::Burn, D> {
    type Element = Tensor<D, Int>;

    fn sample(&self, n: usize) -> Self::Element {
        let mut shape = self.shape.clone();
        shape[0] = n;
        Tensor::random(shape, burn::tensor::Distribution::Uniform(0., 255.), (&self.inner, burn::tensor::DType::U8))
    }

    fn device(&self) -> Device {
        self.inner.clone()
    }
}

impl<const D: usize> ByteSpace<crate::core::Native, D> {
    /// create a new byte space, with feature rank (D - 1) with `Native` representation
    /// # Warning
    /// - note that D is (feature rank + 1)
    /// - this is due to the fixed rank of `burn::Tensor`
    /// - Even if it is in `Native` representation, for unity
    pub fn native(shape: &[usize], seed: u64) -> Self {
        Self { shape: shape.try_into().unwrap(), inner: SmallRng::seed_from_u64(seed) }
    }
    /// return the shape
    /// # Warning
    /// - Due to fixed rank of `burn::Tensor`, shape will always have dimension 1 in the first dimension
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<const D: usize> NativeSpace for ByteSpace<crate::core::Native, D> {
    type Element = ArrayD<u8>;

    fn sample(&mut self) -> Self::Element {
        let n_elems = self.shape.iter().product();
        let elems: Vec<u8> = (0..n_elems).into_iter().map(|_| self.inner.random_range(0u8..255u8)).collect();
        let s = ArrayD::from_shape_vec(self.shape(), elems).unwrap();
        s
    }
}

impl<const D: usize> ToBurnSpace for ByteSpace<crate::core::Native, D> {
    type BurnSpace = ByteSpace<crate::core::Burn, D>;

    fn into_burn(self, device: &Device) -> Self::BurnSpace {
        let shape = self.shape;
        Self::BurnSpace::burn(&shape, device)
    }

    fn bind(elems: Vec<Self::Element>, device: &Device) -> <Self::BurnSpace as BurnSpace>::Element {
        assert!(!elems.is_empty(), "ToBurnSpace::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        let n = elems.len();
        let mut shape = elems[0].shape().to_vec(); shape[0] = n;
        let native = elems.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(native, shape), (device, burn::tensor::DType::U8))
    }

    fn unbind(elem: <Self::BurnSpace as BurnSpace>::Element) -> Vec<Self::Element> {
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