import itertools

import gym
import numpy as np

from envs import grid
from envs import meta_exploration


class InstructionWrapper(meta_exploration.InstructionWrapper):
  """Instruction wrapper for CityGridEnv.

  Provides instructions (goal locations) and their corresponding rewards.

  Reward function for a given goal is:
      R(s, a) = -0.1 if s != goal
              = 1    otherwise
  """

  def _instruction_observation_space(self):
    return gym.spaces.Box(
        np.array([0, 0]), np.array([self.width, self.height]), dtype=np.int)

  def _reward(self, instruction_state, action, original_reward):
    del original_reward

    done = False
    reward = -0.1
    if np.array_equal(self.agent_pos, instruction_state.instructions):
      reward = 1
      done = True
    elif action == grid.Action.end_episode:
      reward -= self.steps_remaining * 0.1  # penalize ending the episode
    return reward, done

  def _generate_instructions(self, test=False):
    del test

    goals = [np.array((0, 0)), np.array((8, 8)),
             np.array((0, 8)), np.array((8, 0))]
    goal = goals[self._random.randint(len(goals))]
    return goal

  def render(self, mode="human"):
    image = super().render(mode)
    image.draw_rectangle(self.current_instructions, 0.5, "green")
    image.write_text("Instructions: {}".format(self.current_instructions))
    return image

  def __str__(self):
    s = super().__str__()
    s += "\nInstructions: {}".format(self.current_instructions)
    return s


class NonstationaryInstructionWrapper(InstructionWrapper):
  """Instruction wrapper for non-stationary version of CityGridEnv."""

  def _generate_instructions(self, test=False):
    del test

    goals = [np.array((0, 0)), np.array((14, 14)),
             np.array((0, 14)), np.array((14, 0))]
    goal = goals[self._random.randint(len(goals))]
    return goal

  def _reward(self, instruction_state, action, original_reward):
    del original_reward

    done = False
    reward = -0.05
    if np.array_equal(self.agent_pos, instruction_state.instructions):
      reward = 1
      done = True
    elif action == grid.Action.ride_bus:
      reward = -0.2
    elif action == grid.Action.end_episode:
      reward -= self.steps_remaining * 0.05  # penalize ending the episode
    return reward, done


class CityGridEnv(grid.GridEnv):
  """Defines a city grid with bus stops at fixed locations.

  Upon toggling a bus stop, the agent is teleported to the next bus stop.
  - The environment defines no reward function (see InstructionWrapper for
  rewards).
  - The episode ends after a fixed number of steps.
  - Different env_ids correspond to different bus destination permutations.
  """

  # Location of the bus stops and the color to render them
  _bus_sources = [
      (np.array((4, 5)), "rgb(0,0,255)"),
      (np.array((5, 4)), "rgb(255,0,255)"),
      (np.array((3, 4)), "rgb(255,255,0)"),
      (np.array((4, 3)), "rgb(0,255,255)"),
  ]

  _destinations = [
      np.array((0, 1)), np.array((0, 7)), np.array((8, 1)), np.array((8, 7)),
  ]

  _bus_permutations = list(itertools.permutations(_destinations))

  _height = 9
  _width = 9

  # Optimization: Full set of train / test IDs is large, so only compute it
  # once. Even though individual IDs are small, the whole ID matrix cannot be
  # freed if we have a reference to a single ID.
  _train_ids = None
  _test_ids = None

  def __init__(self, env_id, wrapper, max_steps=20):
    super().__init__(env_id, wrapper, max_steps=max_steps, width=self._width,
                     height=self._height)

  @classmethod
  def instruction_wrapper(cls):
    return InstructionWrapper

  def _env_id_space(self):
    low = np.array([0])
    high = np.array([len(self._bus_permutations)])
    dtype = np.int
    return low, high, dtype

  @classmethod
  def env_ids(cls):
    ids = np.expand_dims(np.array(range(len(cls._bus_permutations))), 1)
    return np.array(ids), np.array(ids)

  def text_description(self):
    return "bus grid"

  def _place_objects(self):
    super()._place_objects()
    self._agent_pos = np.array([4, 4])

    destinations = self._bus_permutations[
        self.env_id[0] % len(self._bus_permutations)]
    for (bus_stop, color), dest in zip(self._bus_sources, destinations):
      self.place(grid.Bus(color, dest), bus_stop)
      self.place(grid.Bus(color, bus_stop), dest)


class MapGridEnv(CityGridEnv):
  """Includes a map that tells the bus orientations."""

  def _observation_space(self):
    low, high, dtype = super()._observation_space()
    # add dim for map
    env_id_low, env_id_high, _ = self._env_id_space()

    low = np.concatenate((low, [env_id_low[0]]))
    high = np.concatenate((high, [env_id_high[0] + 1]))
    return low, high, dtype

  def text_description(self):
    return "map grid"

  def _place_objects(self):
    super()._place_objects()
    self._map_pos = np.array([5, 3])

  def _gen_obs(self):
    obs = super()._gen_obs()
    map_info = [0]
    if np.array_equal(self.agent_pos, self._map_pos):
      map_info = [self.env_id[0] + 1]
    return np.concatenate((obs, map_info), 0)

  def render(self, mode="human"):
    image = super().render(mode=mode)
    image.draw_rectangle(self._map_pos, 0.4, "black")
    return image


class NonstationaryMapGridEnv(MapGridEnv):
  _gym_disable_underscore_compat = True

  """Wrapper to create non-stationary GridEnv."""
  _height = 15
  _width = 15

  _bus_sources = [
      (np.array((7, 8)), "rgb(0,0,255)"),
      (np.array((8, 7)), "rgb(255,0,255)"),
      (np.array((6, 7)), "rgb(255,255,0)"),
      (np.array((7, 6)), "rgb(0,255,255)"),
  ]

  _destinations = [
      np.array((0, 1)), np.array((0, 13)), np.array((14, 1)), np.array((14, 13)),
  ]

  _bus_permutations = list(itertools.permutations(_destinations))

  def __init__(self, env_id=0, wrapper=None, max_steps=20, mode=None):
    super().__init__(env_id, wrapper, max_steps)
    assert mode in ("all-env-ids", "env_id", None)
    self._mode = mode
    self._privileged_info = None
    self.max_steps = max_steps

  @property
  def action_space(self):
    return gym.spaces.Discrete(6)   # remove end_episode action

  def _place_objects(self):
    self._agent_pos = np.array([8, 6])
    self._map_pos = np.array([7, 7])
    self._place_buses()

  def _place_buses(self):
    destinations = self._bus_permutations[
        self.env_id[0] % len(self._bus_permutations)]
    for (bus_stop, color), dest in zip(self._bus_sources, destinations):
      self.place(grid.Bus(color, dest), bus_stop)
      self.place(grid.Bus(color, bus_stop), dest)

  def _reset(self):
    # Define intervals
    self._switch_intervals = []
    self._env_ids = []
    episode_env_ids = []
    total_time = 0
    while total_time < self._max_steps:
      if np.random.rand() < 0.25:
        interval = 2
      else:
        interval = 3
      self._switch_intervals.append(interval)

      all_env_ids = self.env_ids()[0]
      sampled_env_id = all_env_ids[np.random.randint(len(all_env_ids))]
      self._env_ids.append(sampled_env_id)
      episode_env_ids.extend([sampled_env_id[0],] * interval)
      total_time += interval

    if self._mode == "all-env-ids":
      self._privileged_info = np.array(episode_env_ids)[:self._max_steps]
    else:
      self._privileged_info = np.array([episode_env_ids[0],])
    self._interval_index = 0
    self._steps_since_switch = 0
    return super()._reset()

  def _switch_bus_permutation(self, env_id):
    self._env_id = env_id
    self._grid = [[None for _ in range(self.height)]
                  for _ in range(self.width)]
    self._place_buses()

  def _step(self, action):
    if self._steps_since_switch == self._switch_intervals[self._interval_index]:
      self._steps_since_switch = 0
      self._switch_bus_permutation(self._env_ids[self._interval_index])
      if self._mode != "all-env-ids":
        self._privileged_info = np.array([self._env_ids[self._interval_index][0],])
      self._interval_index += 1

    self._steps_since_switch += 1
    return super()._step(action)

  @classmethod
  def instruction_wrapper(cls):
    return NonstationaryInstructionWrapper

  def _privileged_info_space(self):
    if self._mode is None:
      return np.array([0]), np.array([1]), np.int

    if self._mode == "env_id":
      low, high, dtype = self._env_id_space()
    elif self._mode == "all-env-ids":
      low, high, dtype = self._env_id_space()
      low = low.repeat(self._max_steps)
      high = high.repeat(self._max_steps)
    return low, high, dtype

  @property
  def observation_space(self):
    observation_low, observation_high, _ = self._observation_space()
    env_id_low, env_id_high, dtype = self._env_id_space()
    privileged_info_low, privileged_info_high, dtype = self._privileged_info_space()
    data = {
        "observation": gym.spaces.Box(observation_low, observation_high, dtype=dtype),
        "env_id": gym.spaces.Box(env_id_low, env_id_high, dtype=dtype),
        "privileged_info": gym.spaces.Box(privileged_info_low, privileged_info_high, dtype=dtype),
    }
    return gym.spaces.Dict(data)

  def render(self, mode="human"):
    image = super().render(mode)
    image.write_text(f"Privileged Info ({self._mode}): {self._privileged_info}")
    return image