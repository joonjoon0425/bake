//! A logger struct for all loss structs
//! 
use std::collections::HashMap;
use burn::Tensor;

#[derive(Default)]
pub struct Logger {
    logs: Vec<HashMap<&'static str, Tensor<1>>>,
}

pub trait Recordable {
    fn to_record(&self) -> HashMap<&'static str, Tensor<1>>;
}

impl Logger {
    pub fn record(&mut self, loss: &impl Recordable) {
        self.logs.push(loss.to_record());
    }

    pub fn get(&self, n: usize, name: &str) -> Tensor<1> {
        self.logs.get(n).expect("Logger index out of range").get(name).expect(&format!("No record named: {}", name)).clone()
    }

    pub fn mean(&self) -> HashMap<&'static str, f32> {
        if self.logs.is_empty() {
            return HashMap::new();
        }

        let mut sums: HashMap<&'static str, Tensor<1>> = HashMap::new();
        for map in &self.logs {
            for (name, value) in map {
                match sums.get_mut(name) {
                    Some(acc) => *acc = acc.clone() + value.clone(),
                    None => { sums.insert(name, value.clone()); }
                }
            }
        }
        let n = self.logs.len() as f32;
        sums.into_iter()
            .map(|(name, sum)| (name, (sum.mean() / n).into_scalar::<f32>()))
            .collect()
    }

    pub fn clear(&mut self) {
        self.logs.clear();
    }
    // pub fn seek(&self, name: &str, )
}