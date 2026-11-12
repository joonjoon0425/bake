use std::borrow::Cow;
use crate::core::data::dict::{DictLeaf, BatchedDict, key::Key};
use crate::core::data::{BatchOps, Shaped};
use burn::prelude::*;

/// transition struct
#[derive(BatchOps, Debug, Clone)]
pub struct BatchedTransition<Obs, Constraint, Action, Reward>
where 
    Obs: BatchOps,
    Constraint: BatchOps,
    Action: BatchOps,
    Reward: BatchOps,
{
    /// observation (time step `t`)
    pub obs: Obs,
    /// constraint of actions on observation (time step `t`)
    pub constraint: Constraint,
    /// action (time step `t + 1`)
    pub action: Action,
    /// reward (time step `t + 1`)
    pub reward: Reward,
    /// observation after the action (time step `t + 1`)
    pub next_obs: Obs,
    /// constraint of actions on observation after the action (time step `t + 1`)
    pub next_constraint: Constraint,
    
    /// true if next_obs (time step `t + 1`) is terminal state
    pub terminated: Tensor<1, Bool>,
    /// true if next step (time step `t + 1`) is truncated
    pub truncated: Tensor<1, Bool>,

    /// dictionary for additional elements
    pub dict: BatchedDict
}

impl<Obs, Constraint, Action, Reward> BatchedTransition<Obs, Constraint, Action, Reward>
where 
    Obs: BatchOps,
    Constraint: BatchOps,
    Action: BatchOps,
    Reward: BatchOps,
{
    /// return the current device of `BatchedTransition`
    pub fn device(&self) -> Device {
        self.terminated.device()
    }
}

/// additional implementations for `Shaped<BatchedTransition>`
impl<Obs, Constraint, Action, Reward> Shaped<BatchedTransition<Obs, Constraint, Action, Reward>>
where 
    Obs: BatchOps,
    Constraint: BatchOps,
    Action: BatchOps,
    Reward: BatchOps,
{
    /// insert new item, with given string key
    /// # Panics
    /// - when the batch shape does not matchs
    /// # Returns
    /// - the previously saved value if it existed
    pub fn insert_str<T: BatchOps>(&mut self, key: impl Into<Cow<'static, str>>, value: T) -> Result<Option<DictLeaf>, String> {
        assert!(value.check_batch(&self.batch_shape()), "batch shape does not matches"); 
        
        Ok(self.leaf.dict.insert(key, value))
    }

    /// insert new item, with given key struct
    /// # Panics
    /// - when the batch shape does not matchs
    /// # Returns
    /// - the previously saved value if it existed
    pub fn insert<T: BatchOps>(&mut self, key: Key<T>, value: T) -> Result<Option<T>, String> {
        assert!(value.check_batch(&self.bs), "batch shape does not matches"); 
        
        Ok(self.leaf.dict.insert(key.name, value).map(|v| *v.value.downcast::<T>().unwrap()))
    }

    /// change the value of given string key, using a given closure
    pub fn modify_str<T: BatchOps, F: FnOnce(&mut T)>(&mut self, key: &str, f: F) -> Result<(), String> {
        if !self.leaf.dict.map.contains_key(key) {
            return Err(format!("key: `{}` not found. Existing key and each types are: \n{}", key, self.dict.describe_keys()));
        }

        let leaf = self.leaf.dict.map.get_mut(key).unwrap();
        let v = leaf.value.downcast_mut::<T>()
            .ok_or_else(|| format!("type mismatch: expected {}, got {}", std::any::type_name::<T>(), leaf.vtable.type_name))?;
        f(v);
        assert!(v.check_batch(&self.bs), "modify_str: `{key}` no longer matches batch shape {:?}", self.bs);
        Ok(())
    }
    
    /// change the value of given key struct, using a given closure
    pub fn modify<T: BatchOps, F: FnOnce(&mut T)>(&mut self, key: &Key<T>, f: F) -> Result<(), String> {
        if !self.leaf.dict.map.contains_key(&key.name) {
            return Err(format!("key: `{}` not found. Existing key and each types are: \n{}", key.name, self.dict.describe_keys()));
        }

        let leaf = self.leaf.dict.map.get_mut(&key.name).unwrap();
        let v = leaf.value.downcast_mut::<T>()
            .ok_or_else(|| format!("type mismatch: expected {}, got {}", std::any::type_name::<T>(), leaf.vtable.type_name))?;
        f(v);
        assert!(v.check_batch(&self.bs), "modify: `{}` no longer matches batch shape {:?}", key.name, self.bs);
        Ok(())
    }

    /// remove an item, with given string key
    pub fn remove_str<T: BatchOps>(&mut self, key: &str) -> Result<T, String> {
        self.leaf.dict.remove_str(key)
    }

    /// remove an item, with given struct key
    pub fn remove<T: BatchOps>(&mut self, key: &Key<T>) -> Result<T, String> {
        self.leaf.dict.remove(key)
    }

    
}