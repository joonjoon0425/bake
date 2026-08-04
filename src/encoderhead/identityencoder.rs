use crate::encoderhead::*;

#[derive(Module, Debug)]
pub struct IdentityEncoder<B: Backend, const D: usize> {
    backend: PhantomData<B>
}

impl<B: Backend> Encoder<B, 2> for IdentityEncoder<B, 2> {
    type Obs = Tensor<B, 1>;

    fn forward(&self, obs: <Self::Obs as Batchable<B>>::Batched) -> Tensor<B, 2> { obs }
}

