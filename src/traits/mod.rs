use burn::{Tensor, tensor::{Int, backend::Backend}};

pub trait Batchable<B: Backend> : Sized {
    type Batched;

    fn batch(bundle: Vec<Self>, device: &B::Device) -> Self::Batched;
}

impl<B: Backend, const D: usize> Batchable<B> for [f32; D] {
    type Batched = Tensor<B, 2>;

    fn batch(bundle: Vec<Self>, device: &B::Device) -> Self::Batched {
        Tensor::<B, 2>::from_floats::<[[f32; D]; 2]>(*bundle.as_array().unwrap(), device)
    }
}

impl<B: Backend, const D: usize> Batchable<B> for [f64; D] {
    type Batched = Tensor<B, 2>;

    fn batch(bundle: Vec<Self>, device: &B::Device) -> Self::Batched {
        let flat: Vec<f32> = bundle.iter().flatten().map(|&v| v as f32).collect();
        Tensor::<B, 2>::from_floats(flat.as_slice(), device).reshape([bundle.len(), D])
    }
}

impl<B: Backend, const D: usize> Batchable<B> for [i32; D] {
    type Batched = Tensor<B, 2, Int>;

    fn batch(bundle: Vec<Self>, device: &<B>::Device) -> Self::Batched {
        Tensor::<B, 2, Int>::from_ints::<[[i32; D]; 2]>(*bundle.as_array().unwrap(), device)
    }
}

impl<B: Backend, const D: usize> Batchable<B> for [i64; D] {
    type Batched = Tensor<B, 2, Int>;

    fn batch(bundle: Vec<Self>, device: &<B>::Device) -> Self::Batched {
        Tensor::<B, 2, Int>::from_ints::<[[i64; D]; 2]>(*bundle.as_array().unwrap(), device)
    }
}

impl<B: Backend> Batchable<B> for () {
    type Batched = ();

    fn batch(_bundle: Vec<Self>, _device: &<B>::Device) -> Self::Batched {}
}

macro_rules! impl_batchable_tensor {
    ($d:literal => $db:literal) => {
        impl<B: Backend> Batchable<B> for Tensor<B, $d> {
            type Batched = Tensor<B, $db>;

            fn batch(bundle: Vec<Self>, _device: &B::Device) -> Self::Batched {
                Tensor::stack(bundle, 0)
            }
        }
    };
}

impl_batchable_tensor!(1 => 2);
impl_batchable_tensor!(2 => 3);
impl_batchable_tensor!(3 => 4);
impl_batchable_tensor!(4 => 5);
impl_batchable_tensor!(5 => 6);
impl_batchable_tensor!(6 => 7);

