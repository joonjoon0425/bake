//! A basic network implementations for easy customization
//! 

pub mod mlp;
pub use mlp::{
    Mlp,
    MlpDiscreteDuelingQNet,
    MlpDiscreteQNet,
    MlpPolicyNet,
    MlpSeparatedActorCriticNet,
    MlpSharedActorCriticNet,
};

pub mod noisy_mlp;
pub use noisy_mlp::{
    NoisyMlp
};