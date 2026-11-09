//! dictionary typed BatchOps leaf
use std::{any::Any, borrow::Cow, collections::BTreeMap};
use crate::core::data::{Shaped, BatchOps, batch_ops::helper::*};
use super::key::Key;
use super::vtable::BatchOpsVTable;

/// erased `BatchOps` type
pub type ErasedBatchOps = Box<dyn Any>;

/// dictionary leaf
#[derive(Debug)]
pub struct DictLeaf {
    pub(crate) value: ErasedBatchOps,
    pub(crate) vtable: BatchOpsVTable,
}

impl Clone for DictLeaf {
    fn clone(&self) -> Self {
        Self {
            value: (self.vtable.clone)(&self.value),
            vtable: self.vtable.clone()
        }
    }
}

/// dictionary BatchOps type
#[derive(Debug, Clone)]
pub struct BatchedDict {
    pub(crate) map: BTreeMap<Cow<'static, str>, DictLeaf>,
}

impl BatchedDict {
    /// create empty Dict
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new()
        }
    }

    /// insert new item, with given string key
    /// # Returns
    /// - the previously saved value if it existed
    pub fn insert<T: BatchOps>(&mut self, key: impl Into<Cow<'static, str>>, value: T) -> Option<DictLeaf> {
        self.map.insert(key.into(), DictLeaf {
            vtable: BatchOpsVTable::new::<T>(),
            value: Box::new(value)
        })
    }

    /// get an item, with given string key and data type
    pub fn get_str<T: BatchOps>(&self, key: &str) -> Result<&T, String> {
        let leaf = self.map.get(key)
            .ok_or_else(|| format!("key: `{key}` not found. Existing key and each types are: \n{}", self.describe_keys()))?;
        leaf.value.downcast_ref::<T>()
            .ok_or_else(|| format!("type mismatch: expected {}, got {}", std::any::type_name::<T>(), leaf.vtable.type_name))
    }

    /// get an item, with given key struct
    pub fn get<T: BatchOps>(&self, key: &Key<T>) -> Result<&T, String> {
        let leaf = self.map.get(&key.name)
            .ok_or_else(|| format!("key: `{}` not found. Existing key and each types are: \n{}", key.name, self.describe_keys()))?;
        leaf.value.downcast_ref::<T>()
            .ok_or_else(|| format!("type mismatch: expected {}, got {}", std::any::type_name::<T>(), leaf.vtable.type_name))
    }

    /// remove an item, with given string key
    pub fn remove_str<T: BatchOps>(&mut self, key: &str) -> Result<T, String> {
        // check here
        self.get_str::<T>(key)?;
        let leaf = self.map.remove(key).unwrap();
        Ok(*leaf.value.downcast::<T>().unwrap())
    }

    /// remove an item, with given key struct
    pub fn remove<T: BatchOps>(&mut self, key: &Key<T>) -> Result<T, String> {
        // check here
        self.get(key)?;
        let leaf = self.map.remove(&key.name).unwrap();
        Ok(*leaf.value.downcast::<T>().unwrap())
    }

    /// describe the key and its types
    pub fn describe_keys(&self) -> String {
        let mut string = String::new();
        for (s, leaf) in &self.map {
            string += &format!("key: {s}, type: {}\n", leaf.vtable.type_name);
        }
        string
    }
}

impl BatchOps for BatchedDict {
    fn assign_inplace_batch(&mut self, bs: &[usize], dim: isize, idx: usize, src: Self, src_bs: &[usize]) {
        assert!(self.map.iter().len() == src.map.iter().len(), "Dict::assign_inplace_batch: The number of key differs in self and src");
        for ((dst_str, dst_leaf), (src_str, src_leaf)) in self.map.iter_mut().zip(src.map.into_iter()) {
            assert!(*dst_str == src_str, "BatchDict::assign_inplace_batch: ket mismatch: expected {dst_str}, got {src_str}");
            ((dst_leaf.vtable.assign_inplace_batch)(&mut dst_leaf.value, bs, dim, idx, src_leaf.value, src_bs));
        }
    }

    fn autodiff(self) -> Self {
        let mut map = BTreeMap::new(); 
        for (key, leaf) in self.map.into_iter() {
            map.insert(key, DictLeaf {
                value: (leaf.vtable.autodiff)(leaf.value),
                vtable: leaf.vtable
            });
        }
        BatchedDict {
            map,
        }
    }

    fn cat_batch(bs: &[Vec<usize>], dim: isize, dat: Vec<Self>) -> Self {
        // used for checking the total number of values for each keys
        let n = dat.len();
        let mut map = BTreeMap::new();
        let mut split: BTreeMap<Cow<'static, str>, (Vec<ErasedBatchOps>, BatchOpsVTable)> = BTreeMap::new();

        // collect all item in keys
        for d in dat {
            for (s, leaf) in d.map {
                split.entry(s).or_insert((vec![], leaf.vtable)).0.push(leaf.value);
            }
        }

        for (s, (vec, vtable)) in split {
            assert!(n == vec.len(), "Dict::cat_batch: key '{}' present in {} of {} items", s, vec.len(), n);
            let value = (vtable.cat_batch)(bs, dim, vec);
            map.insert(s, DictLeaf { value, vtable } );
        }

        Self {
            map
        }
    }

    fn check_batch(&self, bs: &[usize]) -> bool {
        let mut b = true;
        for (_, leaf) in &self.map {
            b = b && (leaf.vtable.check_batch)(&leaf.value, bs);
            if !b { return false }
        }
        b
    }

    fn select_batch(self, bs: &[usize], dim: isize, idx: burn::prelude::Tensor<1, burn::prelude::Int>) -> Self {
        let mut map = BTreeMap::new();
        for (s, leaf) in self.map {
            map.insert(s, DictLeaf {
                value: ((leaf.vtable.select_batch)(leaf.value, bs, dim, idx.clone())),
                vtable: leaf.vtable
            });
        }

        Self {
            map
        }
    }

    fn slice_batch(self, bs: &[usize], dim: isize, range: std::ops::Range<usize>) -> Self {
        let mut map = BTreeMap::new();
        for (s, leaf) in self.map {
            map.insert(s, DictLeaf {
                value: ((leaf.vtable.slice_batch)(leaf.value, bs, dim, range.clone())),
                vtable: leaf.vtable
            });
        }

        Self {
            map
        }
    }

    fn unbind_batch(self, bs: &[usize], dim: isize) -> Vec<Self> {
        let (_, d) = bs_after_unbind(bs, dim).unwrap();
        let n = bs[d];
        let mut split: BTreeMap<Cow<'static, str>, (Vec<ErasedBatchOps>, BatchOpsVTable)> = BTreeMap::new();
        for (s, leaf) in self.map {
            let v = (leaf.vtable.unbind_batch)(leaf.value, bs, dim);
            split.insert(s, (v, leaf.vtable));
        }

        let mut maps = vec![BatchedDict { map: BTreeMap::new() }; n];
        for (s, (vec, vtable)) in split {
            for (map, value) in maps.iter_mut().zip(vec.into_iter()) {
                map.map.insert(s.clone(), DictLeaf {
                    value,
                    vtable: vtable.clone()
                });
            }
        }
        maps
    }

    fn zeros_like_batch(bs: &[usize], dim: isize, capacity: usize, src: &Self) -> Self {
        let mut map = BTreeMap::new();
        for (s, leaf) in &src.map {
            map.insert(s.clone(), DictLeaf {
                value: (leaf.vtable.zeros_like_batch)(bs, dim, capacity, &leaf.value),
                vtable: leaf.vtable.clone()
            });
        }
        BatchedDict { map }
    }

    fn reshape_batch(self, bs: &[usize], new_bs: &[usize]) -> Self {
        let mut map = BTreeMap::new();
        for (s, leaf) in self.map {
            map.insert(s, DictLeaf {
                value: (leaf.vtable.reshape_batch)(leaf.value, bs, new_bs),
                vtable: leaf.vtable
            });
        }
        BatchedDict { map }
    }
}

impl Shaped<BatchedDict> {
    /// create new empty `BatchDict`, with given batch shape
    pub fn batch_dict(bs: Vec<usize>) -> Self {
        Self {
            bs,
            leaf: BatchedDict::new()
        }
    }

    /// describe the key and its types
    pub fn describe_keys(&self) -> String {
        self.leaf.describe_keys()
    }

    /// insert new item, with given string key
    /// # Panics
    /// - when the batch shape does not matchs
    /// # Returns
    /// - the previously saved value if it existed
    pub fn insert_str<T: BatchOps>(&mut self, key: impl Into<Cow<'static, str>>, value: T) -> Result<Option<DictLeaf>, String> {
        assert!(value.check_batch(&self.bs), "batch shape does not matches"); 
        
        Ok(self.leaf.insert(key, value))
    }

    /// insert new item, with given key struct
    /// # Panics
    /// - when the batch shape does not matchs
    /// # Returns
    /// - the previously saved value if it existed
    pub fn insert<T: BatchOps>(&mut self, key: Key<T>, value: T) -> Result<Option<T>, String> {
        assert!(value.check_batch(&self.bs), "batch shape does not matches"); 
        
        Ok(self.leaf.insert(key.name, value).map(|v| *v.value.downcast::<T>().unwrap()))
    }

    /// change the value of given string key, using a given closure
    pub fn modify_str<T: BatchOps, F: FnOnce(&mut T)>(&mut self, key: &str, f: F) -> Result<(), String> {
        if !self.leaf.map.contains_key(key) {
            return Err(format!("key: `{}` not found. Existing key and each types are: \n{}", key, self.describe_keys()));
        }

        let leaf = self.leaf.map.get_mut(key).unwrap();
        let v = leaf.value.downcast_mut::<T>()
            .ok_or_else(|| format!("type mismatch: expected {}, got {}", std::any::type_name::<T>(), leaf.vtable.type_name))?;
        f(v);
        assert!(v.check_batch(&self.bs), "modify_str: `{key}` no longer matches batch shape {:?}", self.bs);
        Ok(())
    }

    /// remove an item, with given string key
    pub fn remove_str<T: BatchOps>(&mut self, key: &str) -> Result<T, String> {
        self.leaf.remove_str(key)
    }

    /// remove an item, with given struct key
    pub fn remove<T: BatchOps>(&mut self, key: &Key<T>) -> Result<T, String> {
        self.leaf.remove(key)
    }

    /// change the value of given key struct, using a given closure
    pub fn modify<T: BatchOps, F: FnOnce(&mut T)>(&mut self, key: &Key<T>, f: F) -> Result<(), String> {
        if !self.leaf.map.contains_key(&key.name) {
            return Err(format!("key: `{}` not found. Existing key and each types are: \n{}", key.name, self.describe_keys()));
        }

        let leaf = self.leaf.map.get_mut(&key.name).unwrap();
        let v = leaf.value.downcast_mut::<T>()
            .ok_or_else(|| format!("type mismatch: expected {}, got {}", std::any::type_name::<T>(), leaf.vtable.type_name))?;
        f(v);
        assert!(v.check_batch(&self.bs), "modify: `{}` no longer matches batch shape {:?}", key.name, self.bs);
        Ok(())
    }
}