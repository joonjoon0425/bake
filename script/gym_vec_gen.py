"""
Codegen script for Farama Foundation Gymnasium environments 
"""
import gymnasium as gym
import numpy as np
import tomllib
import importlib

def rust_space_strs(space):
    if isinstance(space, gym.spaces.MultiDiscrete):
        return "BatchedDiscreteSpace"
    elif isinstance(space, gym.spaces.Box):
        if isinstance(space.dtype, (np.dtypes.Float32DType, np.dtypes.Float64DType)):
            n = len(space.shape)
            return f"BatchedContinuousSpace<{n + 1}>"
        elif isinstance(space.dtype, np.dtypes.UByteDType):
            n = len(space.shape)
            return f"BatchedByteSpace<{n + 1}>"
        else:
            raise NotImplementedError(f"The given space is Box, but dtype {type(space.dtype)} is not supported currently")
    else:
        raise NotImplementedError(f"bake do not have `BatchedSpace` correspoding to Gymnasium space of type {type(space)} currently")

def load_entry(entry_point: str):
    module, attr = entry_point.split(":")
    return getattr(importlib.import_module(module), attr)


def build(entry: dict):
    env = gym.make_vec(
        id=entry["id"],
        num_envs=16,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.DISABLED},
        **entry.get("kwargs", {})
    )
    for w in entry.get("wrappers", []):
        env = load_entry(w["entry_point"])(env, **w.get("kwargs", {}))
    return env


toml_bin = open("script/envs_gen.toml", "rb")
gen_src = open("crates/bake-rl/src/env/gym/generated.rs", "w")
d = tomllib.load(toml_bin)

output = """//! DO NOT EDIT
//! Auto-generated code with script/gym_gen.py
//! For regeneration, run following bash script in <path-to-bake> :`uv run script/gym_gen.py`

use crate::env::gym::env::BatchedGymnasiumEnvironmentConfig;
use crate::core::env::space::{BatchedDiscreteSpace, BatchedContinuousSpace, BatchedByteSpace};

"""

for env_name, env_info in d.items():
    # build the environment
    env = build(env_info)
    spec = env.spec.to_json()
    spec_name = env_name.upper() + "_SPEC"
    impl_str = ""
    try:
        obs_space_str = rust_space_strs(env.observation_space)
        action_space_str = rust_space_strs(env.action_space)
        impl_str += \
f"""impl BatchedGymnasiumEnvironmentConfig<{obs_space_str}, {action_space_str}> {{
    /// `{env_info['id']}` environment
    /// - `ObsSpace`: `{obs_space_str}`
    /// - `ActionSpace`: `{action_space_str}`
    pub fn {env_name}() -> Self {{
        Self::new({spec_name})
    }}
}}
"""
        impl_str += \
f"""const {spec_name}: &'static str = r#"{spec}"#;

"""

    except NotImplementedError as err:
        impl_str = f"""
// {env_name} not implemented due to following reason: {err.args}

"""
    output += impl_str

gen_src.write(output)