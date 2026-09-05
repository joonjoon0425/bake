use std::time::Instant;

use bake_deep::{buffer::replay::ReplayBufferConfig, constraint::Unconstrained, data::Batch};
use burn::prelude::*;

fn bench_push(capacity: usize, n_push: usize) {
    let device = Device::default();
    let mut buffer = ReplayBufferConfig::uniform(0, capacity);

    let obs = Tensor::<2>::zeros([1, 4], &device);
    let action = Tensor::<1, Int>::zeros([1], &device);
    let reward = Tensor::<1>::zeros([1], &device);
    let make = || Batch {
        obss: obs.clone(), actions: action.clone(), rewards: reward.clone(),
        next_obss: obs.clone(), constraints: Unconstrained, next_constraints: Unconstrained,
        terminated: reward.clone(), truncated: reward.clone(), extras: (),
    };

    // 첫 push는 lazy init이라 따로 빼둠
    buffer.push(make());

    let t0 = Instant::now();
    for _ in 0..n_push {
        buffer.push(make());
    }
    let dt = t0.elapsed();

    println!("capacity {:>9}: {:>8.2?} total, {:>8.1} ns/push",
             capacity, dt, dt.as_nanos() as f64 / n_push as f64);
}

fn main() {
    let device = Device::cpu();
    for capacity in [10_000usize, 100_000, 1_000_000] {
        let mut store = Tensor::<2>::zeros([capacity, 4], &device);
        let row = Tensor::<2>::zeros([1, 4], &device);
        let n = 200_000;
        let t0 = std::time::Instant::now();
        for i in 0..n {
            let idx = i % capacity;
            let s = std::mem::replace(&mut store, Tensor::empty([0, 0], &device));
            store = s.slice_assign(idx..idx + 1, row.clone());
        }
        let dt = t0.elapsed();
        println!("{capacity}: {:.1} ns/push", dt.as_nanos() as f64 / n as f64);
    }

    for cap in [10_000, 100_000, 1_000_000] {
        bench_push(cap, 200_000);
    }
}