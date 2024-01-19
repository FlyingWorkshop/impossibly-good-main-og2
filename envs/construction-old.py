import collections
import math
import gym
import gym_miniworld
from gym_miniworld import entity
from gym_miniworld import miniworld
from gym_miniworld import random
import torch
import numpy as np
from PIL import Image

from . import meta_exploration, privileged, policy, render


class VaryingTextFrame(entity.TextFrame):
  """A non-static TextFrame."""

  @property
  def is_static(self):
    return False

  def set_text(self, text):
    self.str = text
    self.randomize(None, None)


class ConstructionEnv(miniworld.MiniWorldEnv):
  """The sign environment from IMPORT.

  Touching either the red or blue box ends the episode.
  If the box corresponding to the color_index is touched, reward = +1
  If the wrong box is touched, reward = -1.

  The sign behind the wall either says "blue" or "red"
  """
  _num_rows = 3

  def __init__(self, size=10, max_episode_steps=30, row_index=0):
    params = gym_miniworld.params.DEFAULT_PARAMS.no_random()
    params.set('forward_step', 0.8)
    params.set('turn_step', 45)  # 45 degree rotation

    self._size = size
    self._row_index = row_index
    self._blocked_indices = [
        i for i in range(self._num_rows) if i != row_index]
    
    self._np_random = np.random.RandomState(0)

    super().__init__(params=params, max_episode_steps=max_episode_steps, domain_rand=False)

    # Allow for left / right / forward + custom end episode
    self.action_space = gym.spaces.Discrete(self.actions.move_forward + 1)    # no end episode action

  def set_np_random(self, np_random):
    self._np_random = np_random

  def set_row_index(self, row_index):
    self._row_index = row_index
    self._blocked_indices = [
        i for i in range(self._num_rows) if i != row_index]

  @property
  def steps_remaining(self):
    """Returns the number of timesteps remaining in the episode (int)."""
    return self.max_episode_steps - self.step_count
  
  def _gen_world(self):
    start_room = self.add_rect_room(
        min_x=0,
        max_x=0.3 * self._size,
        min_z=0,
        max_z=self._size * 0.8,
        floor_tex="asphalt",
        no_ceiling=True,
        wall_tex='brick_wall')
    end_room = self.add_rect_room(
        min_x=0.75 * self._size,
        max_x=1.05 * self._size,
        min_z=0,
        max_z=self._size * 0.8,
        floor_tex="asphalt",
        no_ceiling=True,
        wall_tex='brick_wall')
    first_route = self.add_rect_room(
        min_x=0.3 * self._size,
        max_x=0.75 * self._size,
        min_z=0,
        max_z=self._size * 0.3,
        floor_tex="asphalt",
        no_ceiling=True,
        wall_tex='brick_wall')
    second_route = self.add_rect_room(
        min_x=0.3 * self._size,
        max_x=0.75 * self._size,
        min_z=self._size * 0.5,
        max_z=self._size * 0.8,
        floor_tex="asphalt",
        no_ceiling=True,
        wall_tex='brick_wall')

    # Connect left road
    self.connect_rooms(
        start_room, first_route, min_z=0, max_z=self._size * 0.3)
    self.connect_rooms(
        start_room, second_route, min_z=self._size * 0.5,
        max_z=self._size * 0.8)

    # Connect right road
    self.connect_rooms(
        end_room, first_route, min_z=0, max_z=self._size * 0.3)
    self.connect_rooms(
        end_room, second_route, min_z=self._size * 0.5,
        max_z=self._size * 0.8)

    self._objects = []

    self._signs = []
    for i in range(self._num_rows):
      sign = VaryingTextFrame(
          pos=[0.35 * self._size, 1.35, self._size * 0.5 * i],
          dir=-np.pi / 2,
          str=" ",
          height=1,
      )
      self.entities.append(sign)
      self._signs.append(sign)

    self._test = False
    if self._test and self._row_index == 0:
      min_z = 0.64 * self._size
      max_z = 0.66 * self._size
    elif self._test and self._row_index == 1:
      min_z = 0.14 * self._size
      max_z = 0.16 * self._size
    else:
      if self._np_random.rand() < 0.5:
        # Start on the top left corner
        min_z = 0.14 * self._size
        max_z = 0.16 * self._size
      else:
        min_z = 0.64 * self._size
        max_z = 0.66 * self._size

    self.place_agent(
        dir=0,
        min_x=0.14 * self._size,
        max_x=0.16 * self._size,
        min_z=min_z,
        max_z=max_z)

    # Decide when the blocked roads will be blocked
    self._num_intervals = 6
    self._interval_length = self.max_episode_steps // self._num_intervals
    self._blocked_indices_by_interval = []
    blocked_indices = []
    for interval in range(self._num_intervals):
      new_blocked_indices = []
      for i in self._blocked_indices:
        if self._np_random.rand() < 0.7:
          new_blocked_indices.append(i)

      for i in new_blocked_indices:
        if i not in blocked_indices:
          blocked_indices.append(i)
      self._blocked_indices_by_interval.append(list(blocked_indices))

    self._spawned = [False,] * self._num_rows

    # Set privileged info
    self._valid_indices = []
    for i in range(self._num_rows):
      if i in self._blocked_indices_by_interval[0]:
        self._valid_indices.append(0)
      else:
        self._valid_indices.append(1)

  def _gen_world_old(self):
    start_room = self.add_rect_room(
        min_x=0,
        max_x=0.3 * self._size,
        min_z=0,
        max_z=self._size * 1.3,
        floor_tex="asphalt",
        no_ceiling=True,
        wall_tex='brick_wall')
    end_room = self.add_rect_room(
        min_x=0.75 * self._size,
        max_x=1.05 * self._size,
        min_z=0,
        max_z=self._size * 1.3,
        floor_tex="asphalt",
        no_ceiling=True,
        wall_tex='brick_wall')
    first_route = self.add_rect_room(
        min_x=0.3 * self._size,
        max_x=0.75 * self._size,
        min_z=0,
        max_z=self._size * 0.3,
        floor_tex="asphalt",
        no_ceiling=True,
        wall_tex='brick_wall')
    second_route = self.add_rect_room(
        min_x=0.3 * self._size,
        max_x=0.75 * self._size,
        min_z=self._size * 0.5,
        max_z=self._size * 0.8,
        floor_tex="asphalt",
        no_ceiling=True,
        wall_tex='brick_wall')
    third_route = self.add_rect_room(
        min_x=0.3 * self._size,
        max_x=0.75 * self._size,
        min_z=self._size * 1.0,
        max_z=self._size * 1.3,
        floor_tex="asphalt",
        no_ceiling=True,
        wall_tex='brick_wall')

    # Connect left road
    self.connect_rooms(
        start_room, first_route, min_z=0, max_z=self._size * 0.3)
    self.connect_rooms(
        start_room, second_route, min_z=self._size * 0.5,
        max_z=self._size * 0.8)
    self.connect_rooms(
        start_room, third_route, min_z=self._size * 1.0,
        max_z=self._size * 1.3)

    # Connect right road
    self.connect_rooms(
        end_room, first_route, min_z=0, max_z=self._size * 0.3)
    self.connect_rooms(
        end_room, second_route, min_z=self._size * 0.5,
        max_z=self._size * 0.8)
    self.connect_rooms(
        end_room, third_route, min_z=self._size * 1.0,
        max_z=self._size * 1.3)

    self._objects = []

    self._signs = []
    for i in range(self._num_rows):
      sign = VaryingTextFrame(
          pos=[0.35 * self._size, 1.35, self._size * 0.5 * i],
          dir=-np.pi / 2,
          str="C",
          height=1,
      )
      self.entities.append(sign)
      self._signs.append(sign)

    # Start on the top left corner
    self.place_agent(
        dir=0,
        min_x=0,
        max_x=0.1 * self._size,
        min_z=0,
        max_z=1.3 * self._size)

    # # Design 1: For each interval, sample blocked and unblocked roads
    # self._num_intervals = 5
    # self._interval_length = self.max_episode_steps // self._num_intervals
    # self._blocked_indices_by_interval = []
    # for _ in range(self._num_intervals):
    #   # blocked_indices = list(set(np.random.randint(low=0, high=3, size=2)))

    #   # Resample with probability 0.25 otherwise keep the same clear_index
    #   if not clear_index or np.random.rand() < 0.25:
    #     clear_index = np.random.randint(low=0, high=3)

    #   blocked_indices = [i for i in range(self._num_rows) if i != clear_index]
    #   self._blocked_indices_by_interval.append(list(blocked_indices))
    #   self._clear_indices_by_interval.append(clear_index)

    # Design 2: Decide when the blocked roads will be blocked
    self._num_intervals = 8
    self._interval_length = self.max_episode_steps // self._num_intervals
    self._blocked_indices_by_interval = []
    blocked_indices = []
    for interval in range(self._num_intervals):
      new_blocked_indices = []
      for i in self._blocked_indices:
        if np.random.rand() < 0.5:
          new_blocked_indices.append(i)

      for i in new_blocked_indices:
        if i not in blocked_indices:
          blocked_indices.append(i)
      self._blocked_indices_by_interval.append(list(blocked_indices))

    self._spawned = [False,] * self._num_rows

    # Set the blocked signs
    for row in self._blocked_indices_by_interval[0]:
      self._signs[row].set_text("B")

    # Set privileged info
    self._valid_indices = []
    for i in range(self._num_rows):
      if i in self._blocked_indices_by_interval[0]:
        self._valid_indices.append(0)
      else:
        self._valid_indices.append(1)

  def _spawn_cones_at_row(self, row_index):
    cones = []
    for i in range(6):
      cones.append(
          self.place_entity(
              entity.MeshEnt(mesh_name="cone", height=0.75, static=False),
              pos=np.array([0.75 * self._size, 0,
                  (0.5 * row_index + 0.025 + i * 0.05) * self._size
              ])))
    return [tuple(cones)]

  def _in_row(self, row_index):
    return (self.agent.pos[2] >= self._size * 0.5 * row_index
        and self.agent.pos[2] <= self._size * (0.5 * row_index + 0.3))

  def _change_signs_and_spawn_cones(self):
    # Spawn when agent is halfway down the route
    if self.step_count > 0 and self.step_count % self._interval_length == 0:
      # Reset self._spawned, self._objects, and signs
      for obj_pair in self._objects:
        for obj in obj_pair:
          self.entities.remove(obj)

      self._objects = []
      self._spawned = [False,] * self._num_rows
      for i in range(self._num_rows):
        self._signs[i].set_text("C")

      # Set the blocked signs
      interval_idx = self.step_count // self._interval_length
      for row in self._blocked_indices_by_interval[interval_idx]:
        self._signs[row].set_text("B")

      # Set privileged info
      self._valid_indices = []
      for i in range(self._num_rows):
        if i in self._blocked_indices_by_interval[interval_idx]:
          self._valid_indices.append(0)
        else:
          self._valid_indices.append(1)

    # Spawn agent if it is halfway down a blocked route
    interval_idx = self.step_count // self._interval_length
    for row in self._blocked_indices_by_interval[interval_idx]:
      if self._spawned[row]:
        continue
      if self._in_row(row) and self.agent.pos[0] >= 0.4 * self._size:
        self._objects.extend(self._spawn_cones_at_row(row))
        self._spawned[row] = True

  def step(self, action):
    self._change_signs_and_spawn_cones()
    obs, reward, done, info = super().step(action)

    reward = -0.05
    if action == self.actions.move_forward + 1:  # custom end episode action
      done = True
      reward -= self.steps_remaining * 0.05  # penalize ending the episode

    for object_pair in self._objects:
      for obj in object_pair:
        if self.near(obj):
          done = True
          reward = -1.0

    if self.agent.pos[0] >= 0.8 * self._size:
      done = True
      reward = 1.0

    return obs, reward, done, done, info

  def demo_policy(self):
    valid_rows = [self._row_index,]
    return DemoPolicy(self, valid_rows)

  def reset(self, **kwargs):
    obs = super().reset(**kwargs)
    return obs, {}


# From:
# https://github.com/maximecb/gym-miniworld/blob/master/pytorch-a2c-ppo-acktr/envs.py
class TransposeImage(gym.ObservationWrapper):
  def __init__(self, env=None):
    super(TransposeImage, self).__init__(env)
    obs_shape = self.observation_space.shape
    self.observation_space = gym.spaces.Box(
        self.observation_space.low[0, 0, 0],
        self.observation_space.high[0, 0, 0],
        [obs_shape[2], obs_shape[1], obs_shape[0]],
        dtype=self.observation_space.dtype)

  def observation(self, observation):
    return observation.transpose(2, 1, 0)

class PrivilegedMiniWorldEnv(meta_exploration.MetaExplorationEnv):
  _privileged_info_modes = ("env_id", "valid-indices", None)
  _refreshing_modes = ("valid-indices",)

  def __init__(self, env_id, wrapper, max_steps=20, mode=None):
    self.set_mode(mode)
    super().__init__(env_id, wrapper)

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
      return np.array([0]), np.array([1]), int
    raise NotImplementedError

  @classmethod
  def instruction_wrapper(cls):
    return privileged.DummyInstructionWrapper

  @property
  def observation_space(self):
    env_id_low, env_id_high, env_id_dtype = self._env_id_space()
    # privileged_info_low, privileged_info_high, privileged_info_dtype = self._privileged_info_space()
    data = {
        "observation": self._observation_space(),
        "env_id": gym.spaces.Box(env_id_low, env_id_high, dtype=env_id_dtype),
        # "privileged_info": gym.spaces.Box(privileged_info_low, privileged_info_high, dtype=privileged_info_dtype),
    }
    return gym.spaces.Dict(data)


class MiniWorldConstruction(PrivilegedMiniWorldEnv):
  """
  Wrapper around the gym-miniworld Maze conforming to the MetaExplorationEnv
  interface.
  """
  action_cls = miniworld.MiniWorldEnv.Actions

  def __init__(self, env_id, wrapper, mode=None):
    super().__init__(env_id, wrapper, mode=mode)
    self._base_env = ConstructionEnv()
    self._env = TransposeImage(self._base_env)
    self.action_space = self._env.action_space

    # added for ELF
    self._observation_space = self._env.observation_space

  # Grab instance of env and modify it, to prevent creating many envs, which
  # causes memory issues.
  @classmethod
  def create_env(cls, seed, test=False, wrapper=None, mode=None):
    if wrapper is None:
      wrapper = lambda state: torch.tensor(state)

    random = np.random.RandomState(seed)
    train_ids, test_ids = cls.env_ids()
    to_sample = test_ids if test else train_ids
    env_id = to_sample[random.randint(len(to_sample))]
    construction_instance._env_id = env_id
    construction_instance._wrapper = wrapper
    construction_instance.set_mode(mode)
    return construction_instance

  def _gen_privileged_info(self):
    if self._mode is None or self._mode == "env_id":
      return np.array([self.env_id])
    if self._mode == "valid-indices":
      return self._base_env._valid_indices
    raise NotImplementedError

  def _privileged_info_space(self):
    if self._mode is None or self._mode == "env_id":
      return self._env_id_space()
    if self._mode == "valid-indices":
      low = np.array([0,] * self._base_env._num_rows)
      high = np.array([2,] * self._base_env._num_rows)
      return low, high, int
    raise NotImplementedError

  def _step(self, action):
    # Refresh the privileged info
    o = self._env.step(action)
    super()._step(action)
    return o

  def _reset(self):
    # Don't set the seed, otherwise can cheat from initial camera angle position!
    self._env.set_row_index(self.env_id)

    o, _ = self._env.reset()

    # Set the privileged info
    # super()._reset()

    return o

  # def _observation_space(self):
  #   return self._env.observation_space

  @classmethod
  def env_ids(cls):
    return list(range(3)), list(range(3))

  def _env_id_space(self):
    low = np.array([0])
    high = np.array([self._base_env._num_rows])
    dtype = int
    return low, high, dtype

  def render(self, mode="human"):
    first_person_render = self._base_env.render(mode="rgb_array")
    top_render = self._base_env.render(mode="rgb_array", view="top")
    image = render.concatenate(
        [Image.fromarray(first_person_render), Image.fromarray(top_render)],
        "horizontal")
    image.thumbnail((320, 240))
    image = render.Render(image)
    image.write_text("Env ID: {}".format(self.env_id))
    # image.write_text(f"Privileged Info ({self._mode}): {self._privileged_info}")
    image.write_text(f"Signs: {[sign.str for sign in self._base_env._signs]}")
    return image

  def demo_policy(self):
    return self._base_env.demo_policy()


class DemoPolicy(policy.Policy):

  def __init__(self, env, valid_rows):
    self._env = env
    self._valid_rows = valid_rows
    row = self._get_nearest_valid_row()
    self._target = self._make_target(row)
    self._around_corner = False
    self._first_route = None

  def _make_target(self, row):
    return np.array([self._env._size, 0, (0.5 * row + 0.15) * self._env._size])

  def _get_nearest_valid_row(self):
    nearest_dist = np.inf
    nearest_row = None
    pos_z = self._env.agent.pos[2]
    for i in self._valid_rows:
      room = self._env.rooms[2 + i]
      if room.min_z <= pos_z <= room.max_z:
        nearest_row = i
        break
      else:
        dist = np.minimum(np.abs(pos_z - room.min_z), np.abs(pos_z - room.max_z))
        if dist <= nearest_dist:
          nearest_dist = dist
          nearest_row = i
    return nearest_row

  def _get_row(self):
    pos_x = self._env.agent.pos[0]
    pos_z = self._env.agent.pos[2]
    rows = (-1, -1, 0, 1, 2)
    for room, row in zip(self._env.rooms, rows):
      if (room.min_x <= pos_x <= room.max_x) and (room.min_z <= pos_z <= room.max_z):
        return row
    assert False, "Unknown room!"  

  def act(self):
    goal_vec = None
    self._target = self._make_target(self._get_nearest_valid_row())
    if self._get_row() not in self._valid_rows:
      # turn around and walk back
      start_room = self._env.rooms[0]
      target = np.array([np.mean([start_room.max_x, start_room.min_x]), 0, self._env.agent.pos[2]])
      goal_vec = target - self._env.agent.pos

    if goal_vec is None:
      if not self._around_corner:
        corner = np.array([0.3 * self._env._size, 0, self._target[2]])
        goal_vec = corner - self._env.agent.pos
      else:
        goal_vec = self._target - self._env.agent.pos
    goal_dir = goal_vec / np.linalg.norm(goal_vec)

    # More than ~22.5 degrees off in either direction
    if goal_dir.dot(self._env.agent.dir_vec) < 0.9:
      agent_angle = np.arctan2(
          self._env.agent.dir_vec[0], self._env.agent.dir_vec[2])
      goal_angle = np.arctan2(goal_dir[0], goal_dir[2])
      delta = agent_angle - goal_angle
      while delta < 0:
        delta += 2 * math.pi
      if delta <= math.pi:
        return self._env.actions.turn_right, None
      else:
        return self._env.actions.turn_left, None

    # If continuing will hit a wall, rotate away from the wall
    step = (self._env.agent.pos +
            self._env.agent.dir_vec * self._env.params.get_max("forward_step"))
    # Need to test is True for specifically hitting walls
    collision = (self._env.intersect(self._env.agent, step, self._env.agent.radius))
    hit_wall = (collision is True or collision and not isinstance(collision, gym_miniworld.entity.Box))
    if hit_wall:
      updated_dir = self._env.agent.dir + math.pi / 4
      updated_dir_vec = np.array(
          [-math.cos(updated_dir), 0, math.sin(updated_dir)])
      step = (self._env.agent.pos +
              updated_dir_vec * self._env.params.get_max("forward_step"))
      hit_left = self._env.intersect(
          self._env.agent, step, self._env.agent.radius)
      if not hit_left:
        return self._env.actions.turn_left, None
      return self._env.actions.turn_right, None

    if abs(
        self._target[2] - self._env.agent.pos[2]
    ) < 0.1 * self._env._size:
      self._around_corner = True

    return self._env.actions.move_forward, None


# Prevents from opening too many windows.
construction_instance = MiniWorldConstruction(0, None)


class ELFConstructionEnv(MiniWorldConstruction):
  _gym_disable_underscore_compat = True

  def __init__(self, size=10, max_episode_steps=30, row_index=0):
    super().__init__(size, max_episode_steps, row_index)

    # added for ELF
    self._last_reward = None
    self._last_action = None
    self.max_steps = max_episode_steps    
    self._observation_space = {
      "image": self._observation_space,
      "expert": gym.spaces.Box(0, len(self.action_cls), shape=(), dtype=self.action_cls),
      "step": gym.spaces.Box(0, self.max_steps, shape=(), dtype=int)
    }

  @property
  def observation_space(self):
    return self._observation_space

  def _compute_optimal_action(self):
    return self.demo_policy().act()

  def _process_obs(self, obs):
    optimal_action, _ = self._compute_optimal_action()
    obs = {"image": obs, "expert": optimal_action, "step": self._env.step_count}
    return obs
  
  def reset(self, seed=None):
    # self._rng = np.random.RandomState(seed)
    # self._env_id = self.create_env_id(self._rng.randint(1e5))
    # obs = self._reset()

    if seed is not None:
      # seed is only specified when `make_env` is called in train.py
      self._our_seed = seed
    else:
      # reset is called w/o seed in `collect_experiences`, so we should increment by number of procs
      if self._env.step_count > 0:
          self._our_seed += self._num_procs

    # create env_id before placing objects in `super()._reset`
    random = np.random.RandomState(self._our_seed)
    ids, _ = self.env_ids()
    env_id = ids[random.randint(len(ids))]
    self._env_id = env_id
    self._base_env.set_np_random(random)
    obs = self._reset()

    # rendering
    self.last_reward = None
    self.last_action = None
    return self._process_obs(obs), {}

  def step(self, action):
    obs, reward, done, _, info = super()._step(action)
    self.last_action = self.action_cls(action)
    self.last_reward = reward
    self.last_render = self.render()
    return self._process_obs(obs), reward, done, done, info

  def render(self, mode="human"):
    image = super().render()
    optimal_action, _ = self._compute_optimal_action()
    image.write_text("Expert: {}".format(optimal_action.__repr__()))  # optimal next action
    image.write_text("Action: {}".format(self.last_action.__repr__()))  # last action
    image.write_text("Reward: {}".format(self.last_reward))  # last reward
    image.write_text("Timestep: {}".format(self._env.step_count))  # current timestep
    return image