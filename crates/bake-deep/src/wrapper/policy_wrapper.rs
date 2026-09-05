//! A PolicyNet wrapper
//! 
use std::marker::PhantomData;
use burn::module::Module;
use crate::{contract::Policy, distribution::{Distribution, PossibleConstraint}, net::{PolicyNet, layer::NoiseReset}};
/// A PolicyNet wrapper
#[derive(Module, Debug)]
pub struct PolicyWrapper<T: PolicyNet<Params = Dist::Params>, Dist: Distribution> {
    net: T,
    #[module(skip)]
    _p: PhantomData<Dist>,
}

impl<T: PolicyNet<Params = Dist::Params>, Dist: Distribution> PolicyWrapper<T, Dist> {
    /// create a new policy
    pub fn new(net: T) -> Self { Self {net, _p: PhantomData} }
}

impl<T: PolicyNet<Params = Dist::Params>, Dist: Distribution> Policy for PolicyWrapper<T, Dist> {
    type Obs = T::Obs;
    type Dist = Dist;

    fn forward<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> Self::Dist {
        let params = self.net.forward(obs);
        C::create_distribution(params, constraint)
    }
}

impl<T: PolicyNet<Params = Dist::Params> + NoiseReset, Dist: Distribution> NoiseReset for PolicyWrapper<T, Dist> {
    fn reset_noise(&mut self) { self.net.reset_noise(); }
}