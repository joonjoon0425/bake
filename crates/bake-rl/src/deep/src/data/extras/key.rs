//! Keys for ExtraContainer
//! Users are free to implement their own Key type
//!
//! 
use crate::data::extras::Key;
use burn::prelude::*;
/// key for advantage
pub struct Advantage;
impl Key for Advantage {
    type Value = Tensor<1>;
    const NAME: &'static str = "advantage";
}
/// key for log probability
pub struct LogProb;
impl Key for LogProb {
    type Value = Tensor<1>;
    const NAME: &'static str = "log_prob";
}
/// ket for returns
pub struct Return;
impl Key for Return {
    type Value = Tensor<1>;
    const NAME: &'static str = "return";
}