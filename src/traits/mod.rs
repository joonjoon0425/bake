use burn::{Tensor, tensor::backend::Backend};

pub trait Batchable<B: Backend> : Sized {
    type Batched;

    fn batch(bundle: Vec<Self>) -> Self::Batched;
}

impl<B: Backend, const D: usize> Batchable<B> for [f32; D] {
    type Batched = Tensor<B, 2>;

    fn batch(bundle: Vec<Self>) -> Self::Batched {
        
    }
}

macro_rules! impl_batchable_tensor {
    ($d:literal => $db:literal) => {
        impl<B: Backend> Batchable<B> for Tensor<B, $d> {
            type Batched = Tensor<B, $db>;

            fn batch(bundle: Vec<Self>) -> Self::Batched {
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

