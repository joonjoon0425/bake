"""
Automatic code generation for gymnasium[ale_py]
"""
import gymnasium as gym
import utils
import ale_py

def generate_code():
    regis = gym.registry
    
    with_namespace_ids = []

    for ID, spec in regis.items():
        if spec.entry_point == "ale_py.env:AtariEnv":
            version = int(ID.split("-v")[-1])
            if version == 5 and spec.name != "Backgammon" and spec.name != "VideoCheckers" :
                # with namespace
                with_namespace_ids.append(ID)
    
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

    for ID in with_namespace_ids:
        try:
            env = gym.wrappers.FrameStackObservation(gym.wrappers.AtariPreprocessing(gym.make(ID, frameskip=1)), stack_size=4)
            env_name = utils.process_env_name(utils.name_from_id(ID.split("/")[-1]))

            spec = env.spec.to_json()
            spec_name = env_name.upper() + "_SPEC"
            # single env
            obs_space_str = utils.single_rust_space_strs(env.observation_space)
            action_space_str = utils.single_rust_space_strs(env.action_space)
            output += \
f"""impl GymnasiumEnvironmentConfig<{obs_space_str}, {action_space_str}> {{
    /// `{ID}` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `{obs_space_str}`
    /// - `ActionSpace`: `{action_space_str}`
    pub fn {env_name}_preprocessed() -> Self {{
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
    f = open("crates/bake-rl/src/env/gym/env/collection/atari.rs", "w")
    output = generate_code()
    f.write(output)
    f.close()