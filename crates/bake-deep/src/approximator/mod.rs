//! Traits and implementations for all sort of approximators (deep neural networks)

pub mod qfunction;
pub use qfunction::*;

pub mod policy;
pub use policy::*;

pub mod actorcritic;
pub use actorcritic::*;

pub mod wrapper;
pub use wrapper::*;

// pub mod encoder;
// pub use encoder::*;

// pub mod head;
// pub use head::*;

// pub mod composed;
// pub use composed::*;