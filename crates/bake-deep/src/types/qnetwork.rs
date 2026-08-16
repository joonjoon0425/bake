//! A QNetwork trait for value-based methods
use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}, nn::{Linear, LinearConfig, activation::Activation}, tensor::{Device}};
use crate::types::{ActionMask, Batchable};

/// A QNetwork trait for value-based methods
pub trait QNetwork : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;

    fn forward<M: ActionMask<Value = Tensor<2>>>(&self, obs: <Self::Obs as Batchable>::Batched, mask: M, value: f32) -> Tensor<2>;
    fn forward_single<M: ActionMask<Value = Tensor<1>>>(&self, obs: Self::Obs, mask: M, value: f32) -> Tensor<1>;
}

/// A Basic MLP QNetwork Implementation
#[derive(Module, Debug)]
pub struct MLPQNetwork {
    layers: Vec<Linear>,
    activation: Activation,
}

impl MLPQNetwork {
    pub fn new(dims: Vec<usize>, activation: Activation, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MLPQNetwork requires at least two dims: input dimension and output dimension."); }

        let mut layers = Vec::with_capacity(dims.len());
        
        for (i, &dim) in dims[..dims.len() - 1].iter().enumerate() {
            layers.push(LinearConfig::new(dim, dims[i + 1]).init(device))
        }
        
        Self {
            layers,
            activation,
        }
    }
}

impl QNetwork for MLPQNetwork {
    type Obs = Tensor<1>;

    fn forward<M: ActionMask<Value = Tensor<2>>>(&self, obs: <Self::Obs as Batchable>::Batched, mask: M, value: f32) -> Tensor<2> {
        let mut x = obs;
        for layer in self.layers[..self.layers.len() - 1].iter() {
            x = self.activation.forward(layer.forward(x));
        }
        x = self.layers.last().unwrap().forward(x);
        mask.apply(x, value)
    }

    fn forward_single<M: ActionMask<Value = Tensor<1>>>(&self, obs: Self::Obs, mask: M, value: f32) -> Tensor<1> {
        let mut x = obs;
        for layer in self.layers[..self.layers.len() - 1].iter() {
            x = self.activation.forward(layer.forward(x));
        }
        x = self.layers.last().unwrap().forward(x);
        mask.apply(x, value)
    }
}

/// A Dueling Q Network Implementation
#[derive(Module, Debug)]
pub struct DuelingQNetwork<ValueNet: QNetwork, AdvNet: QNetwork<Obs = ValueNet::Obs>> {
    value: ValueNet,
    adv: AdvNet,
}

impl<ValueNet: QNetwork, AdvNet: QNetwork<Obs = ValueNet::Obs>> DuelingQNetwork<ValueNet, AdvNet> {
    pub fn new(value: ValueNet, adv: AdvNet) -> Self {
        Self {
            value,
            adv,
        }
    }
}

impl<ValueNet: QNetwork, AdvNet: QNetwork<Obs = ValueNet::Obs>> QNetwork for DuelingQNetwork<ValueNet, AdvNet> {
    type Obs = ValueNet::Obs;

    fn forward<M: ActionMask<Value = Tensor<2>>>(&self, obs: <Self::Obs as Batchable>::Batched, mask: M, value: f32) -> Tensor<2> {
        let v = self.value.forward(obs.clone(), mask.clone(), value);
        let a = self.adv.forward(obs, mask.clone(), value);
        let mean = mask.clone().mean_dim(1, a.clone());
        mask.apply(v + (a - mean), value)
    }

    fn forward_single<M: ActionMask<Value = Tensor<1>>>(&self, obs: Self::Obs, mask: M, value: f32) -> Tensor<1> {
        let v = self.value.forward_single(obs.clone(), mask.clone(), value);
        let a = self.adv.forward_single(obs, mask.clone(), value);
        let mean = mask.clone().mean_dim(0, a.clone());
        mask.apply(v + (a - mean), value)
    }
}