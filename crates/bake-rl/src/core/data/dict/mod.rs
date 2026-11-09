//! heterogenous dictionary types, with typed keys

mod batched;
pub use batched::{BatchedDict, DictLeaf};
pub use batched::key;