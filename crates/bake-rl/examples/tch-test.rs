use tch::{Tensor, Kind, Device};

pub fn main() {
    let t1 = Tensor::zeros([1, 2, 3], (Kind::Float, Device::Cuda(0)));
    let t2 = Tensor::full([1, 2, 3], 4.5, (Kind::Float, Device::Cuda(0)));
    let t3 = Tensor::ones([1, 2, 3], (Kind::Float, Device::Cuda(0)));

    println!("t1 + t2 = {}", &t1 + &t2);
    println!("t3: {}", t3);
}