use std::any::Any;

#[derive(Debug, Clone)]
pub(crate) struct DictVTable {
    pub(crate) clone: fn(&Box<dyn Any>) -> Box<dyn Any>,
    pub(crate) type_name: &'static str,
}

impl DictVTable {
    pub fn new<T: Clone + 'static>() -> Self {
        Self {
            clone: clone_typed::<T>,
            type_name: std::any::type_name::<T>()
        }
    }
}

fn clone_typed<T: Clone + 'static>(d: &Box<dyn Any>) -> Box<dyn Any> {
    Box::new(d.downcast_ref::<T>().expect("Type mismatch in clone_typed").clone())
}