import gymnasium as gym
import numpy as np
import importlib

def single_rust_space_strs(space):
    """
    space: gymnasium space of single environment
    """
    if isinstance(space, gym.spaces.Discrete):
        return "DiscreteSpace"
    elif isinstance(space, gym.spaces.Box):
        if isinstance(space.dtype, (np.dtypes.Float32DType, np.dtypes.Float64DType)):
            n = len(space.shape)
            return f"ContinuousSpace<{n}>"
        elif isinstance(space.dtype, np.dtypes.UByteDType):
            n = len(space.shape)
            return f"ByteSpace<{n}>"
        else:
            raise NotImplementedError(f"The given space is Box, but dtype {type(space.dtype)} is not supported currently")
    else:
        raise NotImplementedError(f"bake do not have `Space` correspoding to Gymnasium space of type {type(space)} currently")

def vec_rust_space_strs(space):
    """
    space: gymnasium space of single environment (warning: not vector environment)
    """
    if isinstance(space, gym.spaces.Discrete):
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
    """
    entry_point: a string formed like `module name:attribute name`
    - returns attribute `module name`.`attribute name`
    """
    module, attr = entry_point.split(":")
    return getattr(importlib.import_module(module), attr)

def inspect_category(entry_point: str):
    """
    entry_point: attribute `entry_point` of `EnvSpec`
    """
    if isinstance(entry_point, str):
        l = entry_point.split(".")
        return l[2]
    else:
        return " "

def build(entry: dict):
    env = gym.make(entry["id"], **entry.get("kwargs", {}))
    for wrapper in entry.get("wrappers", []):
        env = load_entry(wrapper("entry_point"))(env, **wrapper.get("kwargs", {}))
    return env

def process_env_name(env_name : str):
    # inspect uppercases
    upper_idx = []
    for i, c in enumerate(env_name):
        if c.isupper():
            if i == 0:
                # do not add '_' if uppercase is the first character
                continue
            elif env_name[i - 1].isnumeric():
                # if previous character were a number, do not add '_'
                continue
            upper_idx.append(i)
    for idx in reversed(upper_idx):
        env_name = env_name[:idx] + "_" + env_name[idx:]

    return env_name.lower()

def highest_version(name, ids):
    """
    # Arguments
    - `name`: environment name (without version)
    - `ids`: environment ids

    the function queries in `ids` which version of environment with name `name` is the highest, and returns that version
    # Warning
    the function expects `ids` be of form like `<name>-v<version>`; for example, `CartPole-v1`
    # Notes
    note that `ids` does not has to contain only the environment ids with `name`. It seeks only the environments with name `name`.
    """
    max_version = -1
    for ID in ids:
        env_name, version = ID.split("-v")
        if name == env_name:
            max_version = max(max_version, int(version))
    return max_version

def name_from_id(id: str):
    """
    extract environment name from environment id
    """
    return id.split("-v")[0]