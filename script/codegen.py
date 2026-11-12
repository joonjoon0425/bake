"""
Automatic code generation for gymnasium['category']
"""
import gymnasium as gym
import utils

def generate_code(category: str):
    regis = gym.registry

    ids = []
    names = set()

    for ID, spec in regis.items():
        if utils.inspect_category(spec.entry_point) == category:
            ids.append(ID)
            names.add(spec.name)

    # collect only the highest versions
    highest_version_env_ids = []

    for name in names:
        max_version = utils.highest_version(name, ids)
        highest_version_env_ids.append(f"{name}-v{max_version}")

    # generate code

    output = \
"""//! DO NOT EDIT
//! Auto-generated code with script/codegen.py
//! For regeneration, run following bash script in <path-to-bake> :`uv run script/codegen.py`

#[allow(unused)]
use crate::env::gym::env::{GymnasiumEnvironmentConfig, BatchedGymnasiumEnvironmentConfig};
#[allow(unused)]
use crate::core::env::space::{DiscreteSpace, ContinuousSpace, ByteSpace};
#[allow(unused)]
use crate::core::env::space::{BatchedDiscreteSpace, BatchedContinuousSpace, BatchedByteSpace};

"""
    specs = ""

    for ID in highest_version_env_ids:
        try:
            env = gym.make(ID)
            env_name = utils.process_env_name(utils.name_from_id(ID))

            spec = env.spec.to_json()
            spec_name = env_name.upper() + "_SPEC"
            # single env
            obs_space_str = utils.single_rust_space_strs(env.observation_space)
            action_space_str = utils.single_rust_space_strs(env.action_space)
            output += \
f"""impl GymnasiumEnvironmentConfig<{obs_space_str}, {action_space_str}> {{
    /// `{ID}` environment
    /// - `ObsSpace`: `{obs_space_str}`
    /// - `ActionSpace`: `{action_space_str}`
    pub fn {env_name}() -> Self {{
        Self::new({spec_name})
    }}
}}
"""
            # vec env
            vec_obs_space_str = utils.vec_rust_space_strs(env.observation_space)
            vec_action_space_str = utils.vec_rust_space_strs(env.action_space)
            output += \
f"""impl BatchedGymnasiumEnvironmentConfig<{vec_obs_space_str}, {vec_action_space_str}> {{
    /// Batched `{ID}` environment
    /// - `ObsSpace`: `{vec_obs_space_str}`
    /// - `ActionSpace`: `{vec_action_space_str}`
    pub fn {env_name}() -> Self {{
        Self::new({spec_name})
    }}
}}
"""
            specs += \
f"""
const {spec_name}: &'static str = r#"{spec}"#;

"""
        except NotImplementedError as err:
                output += \
f"""
// {env_name} not implemented due to following reason: {err.args}

"""

    output += specs

    return output

if __name__ == "__main__":
    categories = ["box2d", "mujoco", "classic_control", "toy_text"]
    for category in categories:
        src = open(f"crates/bake-rl/src/env/gym/env/collection/{category}.rs", "w")
        code = generate_code(category)
        src.write(code)
        src.close()