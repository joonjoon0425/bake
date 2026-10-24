mod tensor;
pub use tensor::BatchTensor;

mod empty;
pub use empty::BatchEmpty;

mod ndarray;
pub use ndarray::BatchArray;

mod scalar;

mod macro_test;