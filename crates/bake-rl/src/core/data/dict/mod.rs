//! heterogenous dictionary types, with typed keys

mod batched;
pub use batched::{BatchedDict, BatchedDictLeaf};
pub use batched::batched_key;

mod single;
pub use single::{Dict, DictLeaf};
pub use single::key;