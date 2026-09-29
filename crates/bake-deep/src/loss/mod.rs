//! Traits and struct, enumerations for loss
//! loss_fn module contains an enum `LossFn` which is an enumeration of ordinary loss functions such as `MseLoss`, `HuberLoss` and `SmoothL1Loss`.

pub mod loss_fn;
pub use loss_fn::LossFn;

pub mod traits;