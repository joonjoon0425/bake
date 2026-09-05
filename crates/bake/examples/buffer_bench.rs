use burn::prelude::*;
use std::time::Instant;
 
fn main() {
    let capacity: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let device = Device::default();
 
    let mut store = Tensor::<2>::zeros([capacity, 4], &device);
    let row = Tensor::<2>::zeros([1, 4], &device);
 
    // warm up
    for i in 0..1_000 {
        let idx = i % capacity;
        store.inplace(|a| a.slice_assign(idx..idx + 1, row.clone()));
    }
 
    let n = 200_000;
    let t0 = Instant::now();
    for i in 0..n {
        let idx = i % capacity;
        store.inplace(|a| a.slice_assign(idx..idx + 1, row.clone()));
    }
    let dt = t0.elapsed();
    println!("{capacity}: {:.1} ns/push", dt.as_nanos() as f64 / n as f64);
}