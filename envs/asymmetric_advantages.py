import gym
import torch
import numpy as np
from PIL import Image

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

# import meta_exploration
# from envs import privileged
# import render
# from envs.overcooked.scripted_policy import P2_Policy, Action

from . import render, privileged, meta_exploration
from envs.scripted_policy import P2_Policy, Action

class Overcooked1P(object):
  def __init__(self, seed, wrapper):
    self._seed = seed
    self._wrapper = wrapper
    self._np_random = np.random.RandomState(seed)
    self._max_steps = 40

    # mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages_tomato2")
    mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages_tomato")
    base_env = OvercookedEnv.from_mdp(mdp, horizon=self._max_steps, info_level=0)
    self._env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
    self.action_space = self._env.action_space
    self.observation_space = gym.spaces.Box(
        low=0, high=9, shape=(7,), dtype=int)

    self._steps = 0
    self._num_served_dishes = 0
    self._last_obs = None
    self._pot_state = np.array([5, 2, 3])
    self._default_pot_info = np.array([3, 4, 4])

  def get_p1_obs(self):
    position = self._last_obs["both_agent_obs"][0][-2:]
    direction = np.argmax(self._last_obs["both_agent_obs"][0][:4])

    # Convert object_held from one-hot to integer
    if np.any(self._last_obs["both_agent_obs"][0][4:8]):
      object_held = np.argmax(self._last_obs["both_agent_obs"][0][4:8])
    else:
      object_held = 4

    p1_obs = np.concatenate((
        position,
        [direction],
        [object_held],
    ))

    pot_info = self._default_pot_info.copy()
    # If the agent is at the pot, update the pot info
    if np.array_equal(position, self._pot_state[:2]) and direction == self._pot_state[2]:
      # Convet pot_status from one-hot to integer
      if np.any(self._last_obs["both_agent_obs"][0][25:27]):
        pot_status = np.argmax(self._last_obs["both_agent_obs"][0][25:27])
      else:
        pot_status = 2
      num_onions = self._last_obs["both_agent_obs"][0][27]
      num_tomatoes = self._last_obs["both_agent_obs"][0][28]
      pot_info = np.array([pot_status, num_onions, num_tomatoes])

    p1_obs = np.concatenate((p1_obs, pot_info)).astype(int)
    return p1_obs

  def step(self, action):
    self._steps += 1

    self._env_id = self._p2_policy.get_type()
    p2_action = self._p2_policy.act(self._last_obs)

    if action == Action.end_episode:
      # End episode action
      rew = (self.steps_remaining + 1) * -0.01
      return self.get_p1_obs(), rew, True, {}

    obs, rew, done, info = self._env.step([action, p2_action])

    cooking_bonus = 0.0
    prev_cooking = self._last_obs["both_agent_obs"][0][25]
    curr_cooking = obs["both_agent_obs"][0][25]
    if not prev_cooking and curr_cooking:
      cooking_bonus = 0.2
    self._last_obs = obs

    if rew > 0:
      self._num_served_dishes += 1
    rew = int(rew > 0) + cooking_bonus - 0.01

    if self._steps == self._max_steps or self._num_served_dishes == 2:
      done = True
    return self.get_p1_obs(), rew, done, info

  def reset(self):
    self._steps = 0
    self._num_served_dishes = 0

    self._p2_policy = P2_Policy(np_random=self._np_random)
    self._env_id = self._p2_policy.get_type()

    self._last_obs = self._env.reset()
    return self.get_p1_obs()

  def render(self):
    return self._env.render()

  @property
  def steps_remaining(self):
    """Returns the number of timesteps remaining in the episode (int)."""
    return self._max_steps - self._steps


class PrivilegedOvercookedEnv(meta_exploration.MetaExplorationEnv):
  _privileged_info_modes = ("env_id", None)
  _refreshing_modes = ("env_id", None)

  def __init__(self, env_id, wrapper, max_steps=20, mode=None):
    self._wrapper = wrapper
    self.set_mode(mode)

  def set_mode(self, mode=None):
    self._refresh = mode in self._refreshing_modes
    self._mode = mode
    self._privileged_info = None

  def _step(self, action):
    if self._refresh:
      self._privileged_info = self._gen_privileged_info()

  def _reset(self):
    self._privileged_info = self._gen_privileged_info()

  def _gen_privileged_info(self):
    if self._mode is None:
      return 0
    raise NotImplementedError

  def _privileged_info_space(self):
    if self._mode is None:
      return np.array([0]), np.array([1]), np.int
    raise NotImplementedError

  @classmethod
  def instruction_wrapper(cls):
    return privileged.DummyInstructionWrapper

  @property
  def observation_space(self):
    env_id_low, env_id_high, env_id_dtype = self._env_id_space()
    privileged_info_low, privileged_info_high, privileged_info_dtype = self._privileged_info_space()
    data = {
        "observation": self._observation_space(),
        "env_id": gym.spaces.Box(env_id_low, env_id_high, dtype=env_id_dtype),
        "privileged_info": gym.spaces.Box(privileged_info_low, privileged_info_high, dtype=privileged_info_dtype),
    }
    return gym.spaces.Dict(data)


class Overcooked(PrivilegedOvercookedEnv):
  action_cls = Action

  def __init__(self, env_id, wrapper, mode=None):
    if wrapper is None:
      wrapper = lambda state: torch.tensor(state)
    super().__init__(env_id, wrapper, mode=mode)
    self._env = Overcooked1P(env_id, wrapper)
    self.action_space = self._env.action_space

  def _gen_privileged_info(self):
    if self._mode is None or self._mode == "env_id":
      return np.array([self.env_id])
    raise NotImplementedError

  def _privileged_info_space(self):
    if self._mode is None or self._mode == "env_id":
      return self._env_id_space()
    raise NotImplementedError

  @property
  def env_id(self):
    return self._env._env_id

  @property
  def _env_id(self):
    return self._env._env_id

  def _step(self, action):
    return self._env.step(action)

  def _reset(self):
    return self._env.reset()

  def _observation_space(self):
    return self._env.observation_space

  @classmethod
  def env_ids(cls):
    return list(range(2)), list(range(2))

  def _env_id_space(self):
    low = np.array([0])
    high = np.array([2])
    dtype = np.int
    return low, high, dtype

  def render(self, mode="human"):
    top_render = self._env.render()
    image = Image.fromarray(top_render)
    image.thumbnail((640, 480))
    image = render.Render(image)
    image.write_text("Env ID: {}".format(self.env_id))
    image.write_text(f"Observation: {self._env.get_p1_obs()}")
    image.write_text(f"Privileged Info ({self._mode}): {self._privileged_info}")
    return image

  @property
  def steps_remaining(self):
    return self._env.steps_remaining
  

class ELFOvercooked(Overcooked):
  _gym_disable_underscore_compat = True
  
  def __init__(self, env_id=0, wrapper=None, max_steps=40, mode=None):
    super().__init__(env_id, wrapper, mode)
    
    self.obs_len = 7
    
    # added for ELF
    self._last_reward = None
    self._last_action = None
    self._rng = None
    self.max_steps = self._max_steps