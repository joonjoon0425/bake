use burn::{Tensor, backend::{NdArray, ndarray::NdArrayDevice::Cpu}, nn::LinearConfig, tensor::{Distribution, backend::Backend}};
use burn_autodiff::Autodiff;

type Engine = NdArray;
type AutodiffEngine = Autodiff<Engine>;

pub fn main() {
    let device = Cpu;
    Engine::seed(&device, 12);
    AutodiffEngine::seed(&device, 12);
    let a = LinearConfig::new(4, 2).with_initializer(burn::nn::Initializer::Uniform { min: -1.0, max: 1.0 }).init::<AutodiffEngine>(&device);
    //AutodiffEngine::seed(&device, 12);
    let b = LinearConfig::new(4, 2).with_initializer(burn::nn::Initializer::Uniform { min: -1.0, max: 1.0 }).init::<AutodiffEngine>(&device);
    println!("{:?}", a.weight);  // 두 값 비교
    println!("{:?}", b.weight);

    Engine::seed(&device, 12);
    let a: Tensor<Engine, 1> = Tensor::random([4], Distribution::Default, &device);
    Engine::seed(&device, 12);
    let b: Tensor<Engine, 1> = Tensor::random([4], Distribution::Default, &device);
    println!("{:?}\n{:?}", a.to_data(), b.to_data());
}