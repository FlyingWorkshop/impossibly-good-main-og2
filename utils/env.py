import gym
import gym_minigrid

from envs.env_wrappers import ProcessFrame

def make_env(env_key, num_procs, seed=None):
    env = gym.make(env_key, disable_env_checker=True)
    env._num_procs = num_procs
    if 'vizdoom' in env_key.lower():
        env = ProcessFrame(env, 84, 84)
    if 'Map' in env_key:
        wrapper = env.instruction_wrapper()
        env = wrapper(env, seed=seed)
    env.reset(seed=seed)
    return env
