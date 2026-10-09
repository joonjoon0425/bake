//! traits for raw observations and raw actions
//! - This is only for non-deep environments, since the custom deep RL environment will be implemented with `VectorizedEnvironment` trait.
//! - However, if the user wants to develop a non-deep environment which is convertible to deep rl environment with `SynchronizedEnvironment`,
//!   `BatchableElement` trait must be implemented, since it indicates how the conversion of observation and action must happen.

use bake_rl_core::constraint::{DiscreteMask, Unconstrained};
use bake_rl_core::deep::data::Batchable;
use burn::prelude::*;

/// Trait for elements which can be turned into batchable
pub trait BatchableElement: Clone {
    /// the type of shape of the element
    type Shape;
    /// the shape of element
    const SHAPE: Self::Shape;
    /// Converted batch type
    type Batched: Batchable;
    /// convert the element into batchable type
    fn to_batchable(self, device: &burn::prelude::Device) -> Self::Batched;
    /// convert the batchable type into batch
    fn to_raw(batch: Self::Batched) -> Self; 
}

impl BatchableElement for usize {
    type Shape = [usize; 0];
    const SHAPE: Self::Shape = [];
    type Batched = Tensor<1, Int>;

    fn to_batchable(self, device: &burn::prelude::Device) -> Self::Batched {
        Tensor::<1, Int>::from_ints([self], device)
    }

    fn to_raw(batch: Self::Batched) -> Self {
        batch.into_scalar::<u64>() as usize
    }
}

impl<const D: usize> BatchableElement for [f32; D] {
    type Shape = [usize; 1];
    const SHAPE: Self::Shape = [D];
    type Batched = Tensor<2>;
    fn to_batchable(self, device: &burn::prelude::Device) -> Self::Batched {
        Tensor::<1>::from_floats(self, device).unsqueeze_dim(0)
    }

    fn to_raw(batch: Self::Batched) -> Self {
        let vec: Vec<f32> = batch.into_data().try_into_vec().unwrap();
        vec.try_into().unwrap()
    }
}

impl<const D1: usize, const D2: usize> BatchableElement for [[f32; D2]; D1] {
    type Shape = [usize; 2];
    const SHAPE: Self::Shape = [D1, D2];
    type Batched = Tensor<3>;

    fn to_batchable(self, device: &burn::prelude::Device) -> Self::Batched {
        Tensor::<2>::from_floats(self, device).unsqueeze_dim(0)
    }

    fn to_raw(batch: Self::Batched) -> Self {
        debug_assert_eq!(batch.dims(), [1, D1, D2]);
        let vec: Vec<f32> = batch.into_data().try_into_vec().unwrap();
        std::array::from_fn(|i| std::array::from_fn(|j| vec[i * D2 + j]))
    }
}

impl<const D: usize> BatchableElement for Unconstrained<D> {
    type Shape = [usize; 0];
    const SHAPE: Self::Shape = [];
    type Batched = bake_rl_core::deep::constraint::Unconstrained;

    fn to_batchable(self, _: &burn::prelude::Device) -> Self::Batched {
        bake_rl_core::deep::constraint::Unconstrained
    }

    fn to_raw(_: Self::Batched) -> Self {
        Unconstrained
    }
}

impl<const D: usize> BatchableElement for DiscreteMask<D> {
    type Batched = bake_rl_core::deep::constraint::discrete_constraint::DiscreteMask<2>;
    type Shape = [usize; 1];
    const SHAPE: Self::Shape = [D];

    fn to_batchable(self, device: &burn::prelude::Device) -> Self::Batched {
        bake_rl_core::deep::constraint::discrete_constraint::DiscreteMask::<2>(Tensor::<1, Bool>::from_bool(self.0, device).unsqueeze_dim(0))
    }

    fn to_raw(batch: Self::Batched) -> Self {
        let vec: Vec<bool> = batch.0.into_data().try_into_vec().unwrap();
        DiscreteMask(vec.try_into().unwrap())
    }
}