//! A basic network implementations for easy customization
//! 

pub mod mlp;
pub use mlp::{
    Mlp,
    MlpDiscreteDuelingQNet,
    MlpDiscreteQNet,
    MlpPolicy,
    MlpSeparatedActorCritic,
    MlpSharedActorCritic,
};

pub mod noisy_mlp;
pub use noisy_mlp::{
    NoisyMlp,
    NoisyMlpDiscreteDuelingQNet,
    NoisyMlpDiscreteQNet,
    NoisyMlpPolicy,
    NoisyMlpSeparatedActorCritic,
    NoisyMlpSharedActorCritic,
};