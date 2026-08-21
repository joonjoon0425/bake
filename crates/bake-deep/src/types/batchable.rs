//! A Batchable trait
use burn::{Tensor, tensor::{Bool, Device, Int}};

use crate::{constraint::{DiscreteMask, Unconstrained}};

/// Can translate themselves into batch type
pub trait Batchable : Sized + Clone + Send + Sync + 'static {
    fn concat(data: Vec<Self>) -> Self;
    fn select(self, idx: Tensor<1, Int>) -> Self;
    fn batch_size(&self) -> usize;
    fn device(&self) -> Device;
}

impl<const D: usize> Batchable for DiscreteMask<D> {
    fn concat(data: Vec<Self>) -> Self {
        let data: Vec<Tensor<D, Bool>> = data.into_iter().map(|t| t.0).collect();
        DiscreteMask(Tensor::cat(data, 0))
    }

    fn select(self, idx: Tensor<1, Int>) -> Self {
        DiscreteMask(self.0.select(0, idx))
    }

    fn batch_size(&self) -> usize {
        self.0.shape()[0]
    }

    fn device(&self) -> Device {
        self.0.device()
    }
}

impl Batchable for Unconstrained {
    fn concat(_: Vec<Self>) -> Self { Unconstrained }

    fn select(self, _: Tensor<1, Int>) -> Self { Unconstrained }

    fn batch_size(&self) -> usize { 0 }

    fn device(&self) -> Device { Device::default() }
}

impl<const D: usize> Batchable for Tensor<D> {
    fn concat(bundle: Vec<Self>) -> Self {
        Tensor::cat(bundle, 0)
    }

    fn select(self, idx: Tensor<1, Int>) -> Self {
        self.select(0, idx)
    }

    fn batch_size(&self) -> usize {
        self.shape()[0]
    }

    fn device(&self) -> Device {
        self.device()
    }
}

impl<const D: usize> Batchable for Tensor<D, Int> {
    fn concat(bundle: Vec<Self>) -> Self {
        Tensor::cat(bundle, 0)
    }

    fn select(self, idx: Tensor<1, Int>) -> Self {
        self.select(0, idx)
    }

    fn batch_size(&self) -> usize {
        self.shape()[0]
    }

    fn device(&self) -> Device {
        self.device()
    }
}

impl<const D: usize> Batchable for Tensor<D, Bool> {
    fn concat(bundle: Vec<Self>) -> Self {
        Tensor::cat(bundle, 0)
    }

    fn select(self, idx: Tensor<1, Int>) -> Self {
        self.select(0, idx)
    }

    fn batch_size(&self) -> usize {
        self.shape()[0]
    }

    fn device(&self) -> Device {
        self.device()
    }
}

impl Batchable for () {
    fn concat(_bundle: Vec<Self>) -> Self {}
    fn select(self, _: Tensor<1, Int>) -> Self {}
    fn batch_size(&self) -> usize { 0 }
    fn device(&self) -> Device { Device::default() }
}
// macro_rules! impl_batchable_tensor {
//     ($d:literal => $db:literal) => {
//         impl Batchable for Tensor<$d> {
//             type Batched = Tensor<$db>;

//             fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
//                 Tensor::stack(bundle, 0).to_device(device)
//             }
//         }

//         impl Batchable for Tensor<$d, Int> {
//             type Batched = Tensor<$db, Int>;

//             fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
//                 Tensor::stack(bundle, 0).to_device(device)
//             }
//         }

//         impl Batchable for Tensor<$d, Bool> {
//             type Batched = Tensor<$db, Bool>;

//             fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
//                 Tensor::stack(bundle, 0).to_device(device)
//             }
//         }
//     };
// }

// impl_batchable_tensor!(1 => 2);
// impl_batchable_tensor!(2 => 3);
// impl_batchable_tensor!(3 => 4);
// impl_batchable_tensor!(4 => 5);
// impl_batchable_tensor!(5 => 6);
// impl_batchable_tensor!(6 => 7);
// impl_batchable_tensor!(7 => 8);
// impl_batchable_tensor!(8 => 9);

// impl Batchable for f32 {
//     type Batched = Tensor<1>;

//     fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
//         Tensor::from_floats(data.as_slice(), device)
//     }
// }

// impl Batchable for i64 {
//     type Batched = Tensor<1, Int>;

//     fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
//         Tensor::from_ints(data.as_slice(), device)
//     }
// }

// impl Batchable for bool {
//     type Batched = Tensor<1, Bool>;

//     fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
//         Tensor::from_bool(data.as_slice(), device)
//     }
// }

// impl<const D: usize> Batchable for [f32; D] {
//     type Batched = Tensor<2>;

//     fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
//         let n = data.len();
//         let flat: Vec<f32> = data.into_iter().flatten().collect();
//         Tensor::from_data(TensorData::new(flat, [n, D]), device)
//     }
// }

// impl<const D: usize> Batchable for [i64; D] {
//     type Batched = Tensor<2, Int>;

//     fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
//         let n = data.len();
//         let flat: Vec<i64> = data.into_iter().flatten().collect();
//         Tensor::from_data(TensorData::new(flat, [n, D]), device)
//     }
// }
