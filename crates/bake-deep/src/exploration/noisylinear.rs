//! A NoisyLinear Layer for NoisyNet Agents
//! 

use burn::{Tensor, module::{Module, Param}, tensor::{Device, Distribution, TensorData, linalg::outer, module::linear}};

/// Implementes a Factorised Noisy Linear Layer
#[derive(Module, Debug)]
pub struct NoisyLinear {
    weight_mean: Param<Tensor<2>>,
    weight_std: Param<Tensor<2>>,
    bias_mean: Param<Tensor<1>>,
    bias_std: Param<Tensor<1>>,
    #[module(skip)]
    weight_noise: Tensor<2>,
    #[module(skip)]
    bias_noise: Tensor<1>,
}

impl NoisyLinear {
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        let range = 1f64 / (d_input as f64).sqrt();
        let weight_mean = Tensor::random([d_input, d_output], Distribution::Uniform(-range, range), device);
        let bias_mean = Tensor::random([d_output], Distribution::Uniform(-range, range), device);
        let weight_std = Tensor::from_data(TensorData::new(vec![range * 0.5f64; d_input * d_output], [d_input, d_output]), device);
        let bias_std = Tensor::from_data(TensorData::new(vec![range * 0.5f64; d_output], [d_output]), device);

        let input_noise = Tensor::random([d_input], Distribution::Normal(0f64, 1f64), device);
        let output_noise = Tensor::random([d_output], Distribution::Normal(0f64, 1f64), device);
        let input_noise = input_noise.clone().sign() * input_noise.abs().sqrt();
        let output_noise = output_noise.clone().sign() * output_noise.abs().sqrt();

        Self {
            weight_mean: Param::from_tensor(weight_mean),
            weight_std: Param::from_tensor(weight_std),
            bias_mean: Param::from_tensor(bias_mean),
            bias_std: Param::from_tensor(bias_std),
            weight_noise: outer(input_noise, output_noise.clone()),
            bias_noise: output_noise
        }
    }

    pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        linear(
            input,
            self.weight_mean.val() + self.weight_std.val() * self.weight_noise.clone(),
            Some(self.bias_mean.val() + self.bias_std.val() * self.bias_noise.clone())
        )
    }
}

impl NoiseReset for NoisyLinear {
    fn reset_noise(&mut self) {
        let dims = self.weight_noise.dims();
        let device = self.weight_noise.device();
        let input_noise = Tensor::random([dims[0]], Distribution::Normal(0f64, 1f64), &device);
        let output_noise = Tensor::random([dims[1]], Distribution::Normal(0f64, 1f64), &device);

        let input_noise = input_noise.clone().sign() * input_noise.abs().sqrt();
        let output_noise = output_noise.clone().sign() * output_noise.abs().sqrt();

        self.weight_noise = outer(input_noise, output_noise.clone());
        self.bias_noise = output_noise;
    }
}

pub trait NoiseReset {
    /// Reset the noise if required. Since the base implementation is no-op, be careful.
    fn reset_noise(&mut self) {}
}