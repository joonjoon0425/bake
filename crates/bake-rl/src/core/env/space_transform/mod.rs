//! trait for transformation of spaces
use super::space::{Space, BatchedSpace};

/// space transformer for `Space`
pub trait SpaceTransform<Applicant: Space> {
    /// the space after transformation
    type Applied: Space;

    /// return the transformed space
    fn transform_space(&self, applicant: &Applicant) -> Self::Applied;
    /// transform the element
    fn transform_element(&mut self, applicant: &Applicant, element: Applicant::Element) -> <Self::Applied as Space>::Element;
}

/// space transformer for `BatchedSpace`
pub trait BatchedSpaceTransform<Applicant: BatchedSpace> {
    /// the space after transformation
    type Applied: BatchedSpace;

    /// return the transformed space
    fn transform_space(&self, applicant: &Applicant) -> Self::Applied;
    /// transform the element
    fn transform_element(&mut self, applicant: &Applicant, element: Applicant::Element) -> <Self::Applied as BatchedSpace>::Element;
}