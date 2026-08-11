use burn::{Tensor, tensor::{Bool, Int, TensorData, backend::Backend}};

pub trait Batchable : Sized + Clone + std::fmt::Debug + Send + Sync + 'static {
    type Batched<B: Backend> : Sized + Clone + std::fmt::Debug + Send + Sync + 'static;

    fn batch<B: Backend>(bundle: Vec<Self>, device: &B::Device) -> Self::Batched<B>;
}

impl Batchable for f32 {
    type Batched<B: Backend> = Tensor<B, 1>;

    fn batch<B: Backend>(bundle: Vec<Self>, device: &B::Device) -> Self::Batched<B> {
        Tensor::from_floats(bundle.as_slice(), device)
    }
}

impl Batchable for i64 {
    type Batched<B: Backend> = Tensor<B, 1, Int>;

    fn batch<B: Backend>(bundle: Vec<Self>, device: &B::Device) -> Self::Batched<B> {
        Tensor::from_ints(bundle.as_slice(), device)
    }
}

impl<const D: usize> Batchable for [f32; D] {
    type Batched<B: Backend> = Tensor<B, 2>;

    fn batch<B: Backend>(bundle: Vec<Self>, device: &B::Device) -> Self::Batched<B> {
        let n = bundle.len();
        let flat: Vec<f32> = bundle.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(flat, [n, D]), device)
    }
}

impl<const D: usize> Batchable for [f64; D] {
    type Batched<B: Backend> = Tensor<B, 2>;

    fn batch<B: Backend>(bundle: Vec<Self>, device: &B::Device) -> Self::Batched<B> {
        let n = bundle.len();
        let flat: Vec<f32> = bundle.into_iter().flatten().map(|v| v as f32).collect();
        Tensor::from_data(TensorData::new(flat, [n, D]), device)
    }
}

impl<const D: usize> Batchable for [i32; D] {
    type Batched<B: Backend> = Tensor<B, 2, Int>;

    fn batch<B: Backend>(bundle: Vec<Self>, device: &B::Device) -> Self::Batched<B> {
        let n = bundle.len();
        let flat: Vec<i32> = bundle.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(flat, [n, D]), device)
    }
}

impl<const D: usize> Batchable for [i64; D] {
    type Batched<B: Backend> = Tensor<B, 2, Int>;

    fn batch<B: Backend>(bundle: Vec<Self>, device: &<B>::Device) -> Self::Batched<B> {
        let n = bundle.len();
        let flat: Vec<i64> = bundle.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(flat, [n, D]), device)
    }
}

impl<const D: usize> Batchable for [bool; D] {
    type Batched<B: Backend> = Tensor<B, 2, Bool>;

    fn batch<B: Backend>(bundle: Vec<Self>, device: &B::Device) -> Self::Batched<B> {
        let n = bundle.len();
        let flat: Vec<bool> = bundle.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(flat, [n, D]), device)
    }
}

impl Batchable for () {
    type Batched<B: Backend> = ();

    fn batch<B: Backend>(_bundle: Vec<Self>, _device: &B::Device) -> Self::Batched<B> {}
}

// macro_rules! impl_batchable_tensor {
//     ($d:literal => $db:literal) => {
//         impl<B1: Backend> Batchable for Tensor<B1, $d> {
//             type Batched<B: Backend> = Tensor<B, $db>;

//             fn batch<B: Backend>(bundle: Vec<Self>, _device: &B::Device) -> Self::Batched<B1> {
//                 Tensor::stack(bundle, 0)
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

