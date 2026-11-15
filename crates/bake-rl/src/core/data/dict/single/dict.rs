use std::{any::Any, borrow::Cow, collections::BTreeMap};
use crate::core::data::dict::key::Key;

use super::vtable::DictVTable;

/// dict leaf
#[derive(Debug)]
pub struct DictLeaf {
    pub(crate) value: Box<dyn Any>,
    pub(crate) vtable: DictVTable,
}

impl Clone for DictLeaf {
    fn clone(&self) -> Self {
        let value = (self.vtable.clone)(&self.value);
        Self {
            value,
            vtable: self.vtable.clone(),
        }
    }
}

/// heterogeneous container with typed key
#[derive(Debug, Clone)]
pub struct Dict {
    pub(crate) map: BTreeMap<Cow<'static, str>, DictLeaf>,
}

impl Dict {
    /// create new empty `Dict`
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new()
        }
    }

    
    /// insert new item, with given string key
    /// # Returns
    /// - the previously saved value if it existed
    pub fn insert_str<T: Clone + 'static>(&mut self, key: impl Into<Cow<'static, str>>, value: T) -> Option<DictLeaf> {
        self.map.insert(key.into(), DictLeaf {
            vtable: DictVTable::new::<T>(),
            value: Box::new(value)
        })
    }

    /// insert new item, with given struct key
    /// # Returns
    /// - the previously saved value if it existed
    pub fn insert<T: Clone + 'static>(&mut self, key: Key<T>, value: T) -> Option<DictLeaf> {
        self.map.insert(key.name, DictLeaf {
            vtable: DictVTable::new::<T>(),
            value: Box::new(value)
        })
    }

    /// get an item, with given string key and data type
    pub fn get_str<T: Clone + 'static>(&self, key: &str) -> Result<&T, String> {
        let leaf = self.map.get(key)
            .ok_or_else(|| format!("key: `{key}` not found. Existing key and each types are: \n{}", self.describe_keys()))?;
        leaf.value.downcast_ref::<T>()
            .ok_or_else(|| format!("type mismatch: expected {}, got {}", std::any::type_name::<T>(), leaf.vtable.type_name))
    }

    /// get an item, with given key struct
    pub fn get<T: Clone + 'static>(&self, key: &Key<T>) -> Result<&T, String> {
        let leaf = self.map.get(&key.name)
            .ok_or_else(|| format!("key: `{}` not found. Existing key and each types are: \n{}", key.name, self.describe_keys()))?;
        leaf.value.downcast_ref::<T>()
            .ok_or_else(|| format!("type mismatch: expected {}, got {}", std::any::type_name::<T>(), leaf.vtable.type_name))
    }

    /// remove an item, with given string key
    pub fn remove_str<T: Clone + 'static>(&mut self, key: &str) -> Result<T, String> {
        // check here
        self.get_str::<T>(key)?;
        let leaf = self.map.remove(key).unwrap();
        Ok(*leaf.value.downcast::<T>().unwrap())
    }

    /// remove an item, with given key struct
    pub fn remove<T: Clone + 'static>(&mut self, key: &Key<T>) -> Result<T, String> {
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