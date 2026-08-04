use crate::encoderhead::*;

#[derive(Module, Debug)]
pub struct IdentityEncoder<B: Backend, const D: usize> {
    backend: PhantomData<B>
}

impl<B: Backend, const D: usize> Encoder<B, D> for IdentityEncoder<B, D> {
    type Obs = Tensor<B, D>;

    fn forward(&self, obs: Self::Obs) -> Tensor<B, D> { obs }
}