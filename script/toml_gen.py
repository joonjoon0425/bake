import gymnasium as gym
from gymnasium.envs.registration import EnvSpec

def highest_version(regis: dict[str, EnvSpec]):
    versions = {}
    # collect environment names and versions
    for key, spec in regis.items():
        vs = versions.get(spec.name, None)
        if vs == None:
            versions[spec.name] = [spec.version]
        else:
            versions[spec.name].append(spec.version)
    output = {}
    for name, ver_list in versions.items():
        output[name] = max(ver_list)
    return output

def inspect_module(spec: EnvSpec):
    li = spec.entry_point.split(".")
    return li[2]

def process_env_name(env_name : str):
    # inspect uppercases
    upper_idx = []
    for i, c in enumerate(env_name):
        if c.isupper():
            if i == 0:
                continue
            elif env_name[i - 1].isnumeric():
                continue
            upper_idx.append(i)
    for idx in reversed(upper_idx):
        env_name = env_name[:idx] + "_" + env_name[idx:]

    return env_name.lower()


regis = gym.envs.registry
ver = highest_version(regis)

output = "# DO NOT EDIT! auto-generated with script/toml_gen.py\n\n"

for name, v in ver.items():
    if name == "GymV21Environment" or name == "GymV26Environment":
        continue
    output += \
f"""[{process_env_name(name)}]
id = "{name}-v{v}"

"""

import ale_py
import ale_py.roms

# ALE Added
regis = gym.envs.registry

for key, spec in regis.items():
    if spec.namespace == "ALE":
        if spec.name == "Backgammon" or spec.name == "VideoCheckers":
            continue
        env_name = process_env_name(spec.name)
        output += \
f"""[{env_name}_preprocessed]
id = "ale_py:{key}"
kwargs = {{ frameskip = 1 }}
wrappers = [
    {{ entry_point = "gymnasium.wrappers:AtariPreprocessing" }},
    {{ entry_point = "gymnasium.wrappers:FrameStackObservation", kwargs = {{ stack_size = 4 }} }},
]

"""

# ATARI

# ale_envlist = ale_py.roms.get_all_rom_ids()
# for env_name in ale_envlist:
#     env_id = process_env_id_ale(env_name) + "-v5"
#     output += \
# f"""[{env_name}_preprocessed]
# id = "ale_py:ALE/{env_id}"
# kwargs = {{ frameskip = 1 }}
# wrappers = [
#     {{ entry_point = "gymnasium.wrappers:AtariPreprocessing" }},
#     {{ entry_point = "gymnasium.wrappers:FrameStackObservation", kwargs = {{ stack_size = 4 }} }},
# ]

# """

f = open("script/envs_gen.toml", "w")
f.write(output)
f.close()