//! Basic space for non-deep rl environments
/// Basic space trait
pub trait Space {
    /// the type of element of the space
    type Element: 'static + Clone;
}