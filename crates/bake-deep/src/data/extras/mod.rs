//! Key-typed heterogenous storage for per-transition auxillary data with `Batchable` trait<br>
//! Since `burn`'s `Tensor` has compile time fixed rank `D`, the auxillary types  cannot be added with ease;
//! If it was possible, I would have used `HashMap`, like TensorDict.
use std::{any::{Any, TypeId}, collections::HashMap};
use core::ops::Range;
use burn::prelude::*;
use crate::data::Batchable;

/// key trait which key-types must implement.
pub trait Key: 'static {
    /// the type of value which the key holds
    type Value: Batchable;
    /// human readable name for debuging
    const NAME: &'static str;
}

/// the type-erased data
type Erased = Box<dyn Any + Send + Sync>;

/// A v-table for ExtraContainer to implement Batchable
#[derive(Debug, Clone)]
struct BatchOpsVTable {
    pub len: fn(&Erased) -> Option<usize>,
    pub cat: fn(Vec<Erased>) -> Erased,
    pub select: fn(Erased, Tensor<1, Int>) -> Erased,
    pub slice: fn(Erased, Range<usize>) -> Erased,
    pub detach: fn(Erased) -> Erased,
    pub assign_inplace: fn(&mut Erased, Erased, usize),
    pub zeros_like: fn(usize, &Erased, &Device) -> Erased,
    pub to_device: fn(Erased, &Device) -> Erased,
    pub into_autodiff: fn(Erased) -> Erased,
    pub clone: fn(&Erased) -> Erased
}

impl BatchOpsVTable {
    pub fn new<T: Batchable>() -> Self {
        BatchOpsVTable {
            len: len_impl::<T>,
            cat: cat_impl::<T>,
            select: select_impl::<T>,
            slice: slice_impl::<T>,
            detach: detach_impl::<T>,
            assign_inplace: assign_inplace_impl::<T>,
            zeros_like: zeros_like_impl::<T>,
            to_device: to_device_impl::<T>,
            into_autodiff: into_autodiff_impl::<T>,
            clone: clone_impl::<T>,
        }
    }
}

fn len_impl<T: Batchable>(v: &Erased) -> Option<usize> {
    v.downcast_ref::<T>().expect("ExtraContainer type mismatch in len").batch_size()
}

fn cat_impl<T: Batchable>(v: Vec<Erased>) -> Erased {
    let v: Vec<T> = v.into_iter().map(|val| *val.downcast::<T>().expect("ExtraContainer type mismatch in cat")).collect();
    Box::new(T::cat(v))
}

fn select_impl<T: Batchable>(v: Erased, idx: Tensor<1, Int>) -> Erased {
    Box::new(v.downcast::<T>().expect("ExtraContainer type mismatch in select").select(idx))
}

fn slice_impl<T: Batchable>(v: Erased, range: Range<usize>) -> Erased {
    Box::new(v.downcast::<T>().expect("ExtraContainer type mismatch in slice").slice(range))
}

fn detach_impl<T: Batchable>(v: Erased) -> Erased {
    Box::new(v.downcast::<T>().expect("ExtraContainer type mismatch in detach").detach())
}

fn assign_inplace_impl<T: Batchable>(v: &mut Erased, data: Erased, index: usize) {
    let v = v.downcast_mut::<T>().expect("ExtraContainer type mismatch in assign_inplace: v");
    let data = *data.downcast::<T>().expect("ExtraContainer type mismatch in assign_inplace: data");
    v.assign_inplace(data, index);
}

fn zeros_like_impl<T: Batchable>(capacity: usize, data: &Erased, device: &Device) -> Erased {
    let data = data.downcast_ref::<T>().expect("ExtraContainer type mismatch in zeros_like");
    Box::new(T::zeros_like(capacity, data, device))
}

fn to_device_impl<T: Batchable>(v: Erased, device: &Device) -> Erased {
    let v = v.downcast::<T>().expect("ExtraContainer type mismatch in to_device");
    Box::new(v.to_device(device))
}

fn into_autodiff_impl<T: Batchable>(v: Erased) -> Erased {
    let v = v.downcast::<T>().expect("ExtraContainer type mismatch in into_autodiff");
    Box::new(v.into_autodiff())
}

fn clone_impl<T: Batchable>(v: &Erased) -> Erased {
    let v = v.downcast_ref::<T>().expect("ExtraContainer type mismatch in clone").clone();
    Box::new(v)
}

/// type-erased data leaf for `ExtraContainer`
struct ExtraContainerLeaf {
    pub value: Erased,
    pub vtable: BatchOpsVTable,
    pub type_name: &'static str,
}

/// A homogenous storage with struct key types
pub struct ExtraContainer {
    container: HashMap<TypeId, ExtraContainerLeaf>,
}

impl ExtraContainer {
    /// create a new empty ExtraContainer
    pub fn new() -> Self {
        Self {
            container: HashMap::default(),
        }
    }

    /// push a new key-value pair
    /// 
    /// If the map did not have this key present, None is returned.
    /// 
    /// If the map did have this key present, the value is updated, and the old value is returned. The key is not updated, though; this matters for types that can be == without being identical. See the module-level documentation for more.
    pub fn insert<K: Key>(&mut self, value: K::Value) -> Option<K::Value> {
        let ret = self.container.insert(TypeId::of::<K>(), ExtraContainerLeaf{ value: Box::new(value), vtable: BatchOpsVTable::new::<K::Value>(), type_name: K::NAME });
        if ret.is_some() { Some(*ret.unwrap().value.downcast().unwrap()) } else { None }
    }

    /// get a value of given key-type
    pub fn get<K: Key>(&self) -> Option<&K::Value> {
        self.container.get(&TypeId::of::<K>())?.value.downcast_ref()
    }

    /// get a mutable reference of value of given key-type
    pub fn get_mut<K: Key>(&mut self) -> Option<&mut K::Value> {
        self.container.get_mut(&TypeId::of::<K>())?.value.downcast_mut()
    }

    /// remove a value of given key-type and return the removed value
    pub fn remove<K: Key>(&mut self) -> Option<K::Value> {
        Some(*self.container.remove(&TypeId::of::<K>())?.value.downcast::<K::Value>().expect("ExtraContainer type mismatch in remove"))
    }
}

impl Batchable for ExtraContainer {
    fn batch_size(&self) -> Option<usize> {
        self.container.iter().find_map(|(_, v)| (v.vtable.len)(&v.value))
    }

    /// concatanate along batch dimension (dim 0)
    ///
    /// # Panics
    /// - empty `items`
    /// - The items contains different key types
    fn cat(items: Vec<Self>) -> Self {
        if items.is_empty() { panic!("cat called with empty items") }
        let n = items.len();
        let mut bucket: HashMap<TypeId, (Vec<Erased>, BatchOpsVTable, &'static str)> = HashMap::new();

        for item in items {
            for (t, leaf) in item.container {
                bucket.entry(t).or_insert((vec![], leaf.vtable, leaf.type_name)).0.push(leaf.value);
            }
        }

        let mut container = HashMap::new();
        for (t, (v, vtable, type_name)) in bucket {
            assert_eq!(v.len(), n, "ExtraContainer::cat: key `{}` present in {} of {} items", type_name, v.len(), n);
            let v = (vtable.cat)(v);
            container.insert(t, ExtraContainerLeaf { value: v, vtable, type_name });
        }

        Self {
            container
        }
    }

    fn select(self, idx: Tensor<1, Int>) -> Self {
        let mut container = HashMap::new();
        for (typeid, leaf) in self.container {
            let v = (leaf.vtable.select)(leaf.value, idx.clone());
            container.insert(typeid, ExtraContainerLeaf { value: v, vtable: leaf.vtable, type_name: leaf.type_name });
        }
        Self {
            container
        }
    }

    fn slice(self, range: Range<usize>) -> Self {
        let mut container = HashMap::new();
        for (typeid, leaf) in self.container {
            let v = (leaf.vtable.slice)(leaf.value, range.clone());
            container.insert(typeid, ExtraContainerLeaf { value: v, vtable: leaf.vtable, type_name: leaf.type_name });
        }
        Self {
            container
        }
    }

    fn detach(self) -> Self {
        let mut container = HashMap::new();
        for (typeid, leaf) in self.container {
            let v = (leaf.vtable.detach)(leaf.value);
            container.insert(typeid, ExtraContainerLeaf { value: v, vtable: leaf.vtable, type_name: leaf.type_name });
        }
        Self {
            container
        }
    }

    fn assign_inplace(&mut self, data: Self, index: usize) {
        for (typeid, leaf) in data.container {
            (leaf.vtable.assign_inplace)(&mut self.container.get_mut(&typeid).unwrap().value, leaf.value, index)
        }
    }

    fn zeros_like(capacity: usize, data: &Self, device: &Device) -> Self {
        let mut container = HashMap::new();
        for (typeid, leaf) in &data.container {
            let v = (leaf.vtable.zeros_like)(capacity, &leaf.value, device);
            container.insert(*typeid, ExtraContainerLeaf { value: v, vtable: leaf.vtable.clone(), type_name: leaf.type_name });
        }
        Self {
            container
        }
    }

    fn to_device(self, device: &Device) -> Self {
        let mut container = HashMap::new();
        for (typeid, leaf) in self.container {
            let v = (leaf.vtable.to_device)(leaf.value, device);
            container.insert(typeid, ExtraContainerLeaf { value: v, vtable: leaf.vtable, type_name: leaf.type_name });
        }
        Self {
            container
        }
    }

    fn into_autodiff(self) -> Self {
        let mut container = HashMap::new();
        for (typeid, leaf) in self.container {
            let v = (leaf.vtable.into_autodiff)(leaf.value);
            container.insert(typeid, ExtraContainerLeaf { value: v, vtable: leaf.vtable, type_name: leaf.type_name });
        }
        Self {
            container
        }
    }
}

impl Clone for ExtraContainer {
    fn clone(&self) -> Self {
        let mut container = HashMap::new();
        for (typeid, leaf) in &self.container {
            let v = (leaf.vtable.clone)(&leaf.value);
            container.insert(*typeid, ExtraContainerLeaf { value: v, vtable: leaf.vtable.clone(), type_name: leaf.type_name });
        }
        Self {
            container
        }
    }
}

impl std::fmt::Debug for ExtraContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("ExtraContainer");
        for leaf in self.container.values() {
            d.field(leaf.type_name, &(leaf.vtable.len)(&leaf.value));
        }
        d.finish()
    }
}

pub mod key;
pub use key::*;

#[cfg(test)]
mod tests {
    use crate::data::{Batchable, extras::{ExtraContainer, Key, key::*}};
    use burn::prelude::*;
    
    /// test feature
    struct Feature;
    impl Key for Feature { type Value = Tensor<2>; const NAME: &'static str = "test_feature";}

    fn sample(device: &Device) -> ExtraContainer {
        let mut e = ExtraContainer::new();
        e.insert::<Advantage>(Tensor::<1>::from_floats([0.0, 1.0, 2.0, 3.0], device));
        e.insert::<LogProb>(Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], device));
        e.insert::<Return>(Tensor::<1>::from_floats([2.0, 3.0, 4.0, -2.0], device));
        e
    }

    #[test]
    fn init_test() {
        let _extra = ExtraContainer::new();
    }

    #[test]
    fn insert_and_get_test() {
        let mut extra = ExtraContainer::new();
        let device = Device::default();
        extra.insert::<Advantage>(Tensor::<1>::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &device));
        extra.insert::<LogProb>(Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device));
        extra.insert::<Return>(Tensor::<1>::from_floats([2.0, 3.0, 4.0], &device));

        let advantage = extra.get::<Advantage>().unwrap();
        let log_prob = extra.get::<LogProb>().unwrap();
        let ret = extra.get::<Return>().unwrap();

        println!("{:?}", extra);

        println!("{advantage}\n{log_prob}\n{ret}");
    }

    #[test]
    fn mixed_ranks() {
        let device = Device::default();
        let mut e = ExtraContainer::new();
        e.insert::<Advantage>(Tensor::<1>::from_floats([0.0, 1.0], &device));
        e.insert::<Feature>(Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device));

        assert_eq!(e.get::<Feature>().unwrap().shape(), Shape::new([2, 2]));
        let e = e.select(Tensor::<1, Int>::from_ints([1], &device));
        assert_eq!(e.get::<Feature>().unwrap().shape(), Shape::new([1, 2]));
        assert_eq!(e.get::<Advantage>().unwrap().shape(), Shape::new([1]));
    }

    #[test]
    #[should_panic]
    fn cat_with_mismatched_keys_panics() {
        let device = Device::default();
        let mut other = ExtraContainer::new();
        let mut this = ExtraContainer::new();
        other.insert::<Advantage>(Tensor::<1>::from_floats([0.0, 1.0, 2.0, 3.0], &device));
        this.insert::<Return>(Tensor::<1>::from_floats([0.0, 1.0, 2.0, 3.0], &device));
        let _ = ExtraContainer::cat(vec![this, other]);
    }

    #[test]
    fn missing_key_returns_none() {
        let device = Device::default();
        let e = sample(&device);
        assert!(e.get::<Feature>().is_none());
    }

    #[test]
    fn batch_ops_test() {
        let mut extra = ExtraContainer::new();
        let device = Device::default();
        extra.insert::<Advantage>(Tensor::<1>::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &device));
        extra.insert::<LogProb>(Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device));
        extra.insert::<Return>(Tensor::<1>::from_floats([2.0, 3.0, 4.0, -2.0, 11.0, 12.0], &device));


    }
}