use bake::{agent::{dqn::DqnAgent, *}, encoderhead::Head, environment::CartPole};
use burn::{nn::{Linear, LinearConfig, Relu}, prelude::*};
use burn::module::Module;

fn main() {
    let mut env = CartPole::new(123);
    let mut agent = DqnAgent::new(

    );
}

#[derive(Module, Debug)]
struct QHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> QHead<B> {
    pub fn new(d_input: usize, d_output: usize) -> Self {
        LinearConfig {
            
        }
    }
}

impl<B: Backend> Head<B, 2> for QHead<B> {
    type Output = Tensor<B, 1>;

    fn forward(&self, encoded: Tensor<B, 2>) -> Self::Output {
        let x = fc1
    }
}