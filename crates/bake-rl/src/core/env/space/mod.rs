//! Environment spaces
//! 

pub mod discrete;
pub mod continuous;
pub mod dict;

/// Basic space trait
pub trait Space {
    /// the type of element of the space
    type Element: Clone + 'static;
}