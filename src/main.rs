use burn::backend::cuda::CudaDevice;
use burn::backend::ndarray::NdArrayDevice::Cpu;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::nn::loss::{MseLoss, Reduction};

type Engine = burn::backend::NdArray;
type AutoDiffEngine = burn_autodiff::Autodiff<Engine>;

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct MLPConfig {
    d_input: usize,
    d_hidden: usize,
    d_output: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        MLP {
            fc1: LinearConfig::new(self.d_input, self.d_hidden).init(device),
            fc2: LinearConfig::new(self.d_hidden, self.d_output).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        self.fc2.forward(x)
    }
}

fn main() {
    // let device = CudaDevice::new(0);
    let device = Cpu;
    let mut model = MLPConfig::new(1, 32, 1).init::<AutoDiffEngine>(&device);
    let mut optim = AdamConfig::new().init();
    let lr = 1e-3;

    let x = Tensor::<AutoDiffEngine, 2>::random([64, 1], burn::tensor::Distribution::Uniform(-3.0, 3.0), &device);
    let y = x.clone().sin();

    for step in 0..2000 {
        let pred = model.forward(x.clone());
        let loss = MseLoss::new().forward(pred, y.clone(), Reduction::Mean);

        let loss_value = loss.clone().into_scalar();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);

        if step % 200 == 0 {
            println!("step {step}: loss = {loss_value:.6}")
        }
    }

    let model = model.valid();
    let out = model.forward(Tensor::from_data([[0.5]], &device));
    println!("{out}");
}