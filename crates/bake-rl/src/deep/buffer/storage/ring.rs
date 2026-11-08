//! circular, ring storage

use std::borrow::Cow;
use crate::core::data::{Batch, BatchOps, BurnTransition, batch::dict::{DictLeaf, Key}};

/// ring-like storage
/// - wraps over when the pre-allocated storage is full
/// - allocates lazely
/// - used for replay buffer and rollout buffer
#[derive(Debug, Clone)]
pub struct RingStorage<T: BatchOps> {
    head: usize,
    storage: Option<Batch<T>>,
    len: usize,
    full: bool,
}

impl<T: BatchOps> RingStorage<T> {
    /// create new empty, unallocated `RingStorage`
    pub fn new() -> Self {
        Self {
            head: 0,
            len: 0,
            full: false,
            storage: None,
        }
    }

    /// initialize `RingStorage` with given data
    /// # Warning
    /// - the previous storage will be removed if it existed
    pub fn init(&mut self, data: Batch<T>) {
        self.storage = Some(data);
    }

    /// clear `RingStorage`
    pub fn clear(&mut self) {
        self.storage = None;
        self.head = 0;
        self.len = 0;
        self.full = false;
    }

    /// write given data over the existing storage, starting from `self.head` along the first batch dimension
    /// # Warning
    /// - wraps around when capacity is full; but panics if length of data itself is bigger than the capacity
    /// # Panics
    /// - when the storage is not initialized (`self.storage` is `None`)
    /// - when the storage and given data has different batch shape (except the first dimension)
    /// - when the given data has longer length then storage capacity
    /// # Returns
    /// - the indices where the data was written on
    pub fn write(&mut self, data: Batch<T>) -> Vec<usize> {
        assert!(self.storage.is_some(), "RingStorage::write: the storage is not initialized. initialize the storage with RingStorage::init");
        assert!(self.storage.as_ref().unwrap().batch_shape()[1..] == data.batch_shape()[1..], "RingStorage::write: given data and storage has different batch shape");
        let capacity = self.storage.as_ref().unwrap().batch_shape()[0];
        assert!(capacity >= data.batch_shape()[0], "RingStorage::write: given data has longer length than storage's capacity");

        debug_assert!(capacity > self.head, "RingStorage::write: debug_assert: capacity is smaller than self.head. capacity: {}, self.head: {}", capacity, self.head);

        let leftover = data.batch_shape()[0];
        let capability = capacity - self.head;
        if leftover <= capability {
            self.storage.as_mut().unwrap().assign_inplace(0, self.head, data);
            let indices = (self.head..self.head + leftover).collect();
            // move head
            self.head += leftover;
            if !self.full {
                self.len = self.head;
            }
            indices
        } else {
            // divide
            let tail = data.clone().slice(0, 0..capability);
            let head = data.slice(0, capability..leftover);
            let indices = (self.head..capacity).chain(0..leftover - capability).collect();
            self.storage.as_mut().unwrap().assign_inplace(0, self.head, tail);
            self.storage.as_mut().unwrap().assign_inplace(0, 0, head);
            // move head
            self.head = leftover - capability;
            if !self.full {
                self.len = capacity;
                self.full = true;
            }
            indices
        }
    }

    /// return current length of storage
    pub fn len(&self) -> usize {
        self.len
    }

    /// return `true` if the storage is initialized
    pub fn is_init(&self) -> bool {
        self.storage.is_some()
    }

    /// return internal storage
    pub fn storage(&self) -> Option<&Batch<T>> {
        self.storage.as_ref()
    }

    /// convert itself into Batch<T>
    pub fn try_into_batch(self) -> Option<Batch<T>> {
        self.storage
    }

}

impl<Obs: BatchOps, Action: BatchOps, Constraint: BatchOps, Reward: BatchOps> RingStorage<BurnTransition<Obs, Action, Constraint, Reward>> {
    /// insert an extra value with given key
    /// # Panics
    /// - when the storage is not initialized
    pub fn insert<T: BatchOps>(&mut self, key: Key<T>, value: T) -> Result<Option<T>, String> {
        assert!(self.storage.is_some(), "RingStorage::write: the storage is not initialized. initialize the storage with RingStorage::init");
        self.storage.as_mut().unwrap().insert(key, value)
    }

    /// insert an extra value with given key
    /// # Panics
    /// - when the storage is not initialized
    pub fn insert_str<T: BatchOps>(&mut self, key: impl Into<Cow<'static, str>>, value: T) -> Result<Option<DictLeaf>, String> {
        assert!(self.storage.is_some(), "RingStorage::write: the storage is not initialized. initialize the storage with RingStorage::init");
        self.storage.as_mut().unwrap().insert_str(key, value)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::data::*;
    use crate::deep::buffer::storage::ring::RingStorage;
    use burn::prelude::*;

    #[test]
    fn wrap_over() {
        let device = Device::flex();
        let t = Tensor::<1>::from_floats([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], &device);
        let mut buf = RingStorage::new(); buf.init(Batch::new(t, vec![11]));
        let head = Tensor::<1>::from_floats([11., 12., 13., 14.], &device);
        buf.write(Batch::new(head, vec![4]));
        let tail = Tensor::<1>::from_floats([0., 1., 2., 3., 4., 5., 6., 7., 8.], &device);
        let indices = buf.write(Batch::new(tail, vec![9]));
        assert_eq!(indices, vec![4, 5, 6, 7, 8, 9, 10, 0, 1]);
        assert_eq!(buf.storage.unwrap().destruct().0.try_into_vec_as::<f32>().unwrap(), vec![7., 8., 13., 14., 0., 1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn length_before_full() {
        let device = Device::flex();
        let t = Tensor::<1>::from_floats([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], &device);
        let mut buf = RingStorage::new(); buf.init(Batch::new(t, vec![11]));
        assert_eq!(0, buf.len());

        let head = Tensor::<1>::from_floats([11., 12., 13., 14.], &device);
        buf.write(Batch::new(head, vec![4]));

        assert_eq!(4, buf.len());
        assert!(!buf.full);
    }

    #[test]
    fn length_after_full() {
        let device = Device::flex();
        let t = Tensor::<1>::from_floats([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], &device);
        let mut buf = RingStorage::new(); buf.init(Batch::new(t, vec![11]));
        assert_eq!(0, buf.len());

        let head = Tensor::<1>::from_floats([11., 12., 13., 14.], &device);
        buf.write(Batch::new(head, vec![4]));
        assert!(!buf.full);

        let tail = Tensor::<1>::from_floats([0., 1., 2., 3., 4., 5., 6., 7., 8.], &device);
        buf.write(Batch::new(tail, vec![9]));

        assert_eq!(11, buf.len());
        assert!(buf.full);
    }

    #[test]
    #[should_panic]
    fn too_long_data() {
        let device = Device::flex();
        let t = Tensor::<1>::from_floats([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], &device);
        let mut buf = RingStorage::new(); buf.init(Batch::new(t, vec![11]));
        let head = Tensor::<1>::from_floats([11., 12., 13., 14., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], &device);
        buf.write(Batch::new(head, vec![15])); // panic
    }

    #[test]
    #[should_panic]
    fn uninitialized_storage() {
        let device = Device::flex();
        let mut buf = RingStorage::new(); 
        let head = Tensor::<1>::from_floats([11., 12., 13., 14., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], &device);
        buf.write(Batch::new(head, vec![15])); // panic
    }
}