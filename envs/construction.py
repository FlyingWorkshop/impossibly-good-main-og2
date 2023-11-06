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

from . import meta_exploration
from envs import privileged
from . import policy
from . import render


class ConstructionEnv(miniworld.MiniWorldEnv):
  """The sign environment from IMPORT.

  Touching either the red or blue box ends the episode.
  If the box corresponding to the color_index is touched, reward = +1
  If the wrong box is touched, reward = -1.

  The sign behind the wall either says "blue" or "red"
  """

  def __init__(self, size=10, max_episode_steps=40, row_index=0):
    params = gym_miniworld.params.DEFAULT_PARAMS.no_random()
    params.set('forward_step', 0.8)
    params.set('turn_step', 30)  # 30 degree rotation

    self._size = size
    self._row_index = row_index

    super().__init__(
        params=params, max_episode_steps=max_episode_steps, domain_rand=False)

    # Allow for left / right / forward + custom end episode
    self.action_space = gym.spaces.Discrete(self.actions.move_forward + 2)
    self.max_steps = 20

  def set_row_index(self, row_index):
    self._row_index = row_index

  @property
  def steps_remaining(self):
    """Returns the number of timesteps remaining in the episode (int)."""
    return self.max_episode_steps - self.step_count

  def _gen_world(self):
    start_room = self.add_rect_room(
        min_x=0, max_x=0.4 * self._size, min_z=0, max_z=self._size * 1.3,
        floor_tex="asphalt", no_ceiling=True, wall_tex='brick_wall')
    first_route = self.add_rect_room(
        min_x=0.4 * self._size, max_x=self._size, min_z=0,
        max_z=self._size * 0.3, floor_tex="asphalt", no_ceiling=True,
        wall_tex='brick_wall')
    second_route = self.add_rect_room(
        min_x=0.4 * self._size, max_x=self._size, min_z=self._size * 0.5,
        max_z=self._size * 0.8, floor_tex="asphalt", no_ceiling=True,
        wall_tex='brick_wall')
    third_route = self.add_rect_room(
        min_x=0.4 * self._size, max_x=self._size, min_z=self._size * 1.0,
        max_z=self._size * 1.3, floor_tex="asphalt", no_ceiling=True,
        wall_tex='brick_wall')
    self.connect_rooms(
        start_room, first_route, min_z=0, max_z=self._size * 0.3)
    self.connect_rooms(
        start_room, second_route, min_z=self._size * 0.5,
        max_z=self._size * 0.8)
    self.connect_rooms(
        start_room, third_route, min_z=self._size * 1.0,
        max_z=self._size * 1.3)

    self._objects = [
        (self.place_entity(entity.MeshEnt(mesh_name="cone", height=0.75),
            pos=np.array([0.75 * self._size, 0, self._size * 0.5 * self._row_index + 0.025 * self._size])),
         self.place_entity(entity.MeshEnt(mesh_name="cone", height=0.75),
            pos=np.array([0.75 * self._size, 0, self._size * 0.5 * self._row_index + 0.075 * self._size])),
         self.place_entity(entity.MeshEnt(mesh_name="cone", height=0.75),
            pos=np.array([0.75 * self._size, 0, self._size * 0.5 * self._row_index + 0.125 * self._size])),
         self.place_entity(entity.MeshEnt(mesh_name="cone", height=0.75),
            pos=np.array([0.75 * self._size, 0, self._size * 0.5 * self._row_index + 0.175 * self._size])),
         self.place_entity(entity.MeshEnt(mesh_name="cone", height=0.75),
            pos=np.array([0.75 * self._size, 0, self._size * 0.5 * self._row_index + 0.225 * self._size])),
         self.place_entity(entity.MeshEnt(mesh_name="cone", height=0.75),
            pos=np.array([0.75 * self._size, 0, self._size * 0.5 * self._row_index + 0.275 * self._size])),
        )
    ]

    text = ["1", "2", "3"][self._row_index]
    sign = gym_miniworld.entity.TextFrame(
        pos=[0, 1.35, self._size * 0.5 * 1.3],
        dir=0,
        str=text,
        height=1,
    )
    self.entities.append(sign)
    self.place_agent(min_x=0, max_x=0.4 * self._size, min_z=0, max_z=1.3 * self._size)

  def step(self, action):
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

    return obs, reward, done, info

  def demo_policy(self):
    valid_rows = [i for i in range(3) if i != self._row_index]
    return DemoPolicy(self, valid_rows)


class ELFConstructionEnv(ConstructionEnv):
  def __init__(self, size=10, max_episode_steps=40, row_index=0):
    super().__init__(size, max_episode_steps, row_index)

    # transpose observation space to match ELF's Vizdoom models
    h, w, c = self.observation_space.shape
    self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(c, h, w),
            dtype=np.uint8
    )
  
  def reset(self):
    obs = super().reset()
    obs = np.transpose(obs, axes=(2, 0, 1))
    return obs, {}

  def step(self):
    obs, reward, done, info = super().step()
    obs = np.transpose(obs, axes=(2, 0, 1))
    return obs, reward, done, info


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


class PrivilegedMiniWorldEnv(object):
  _privileged_info_modes = ("env_id", None)
  _refreshing_modes = ()

  def __init__(self, env_id, wrapper, max_steps=20, mode=None):
    self._refresh = mode in self._refreshing_modes
    self._mode = mode
    self._privileged_info = None  # IMPORTANT: make sure to call self._gen_privileged_info() in concrete subclasses to avoid rendering bug

  def _gen_privileged_info(self):
    if self._mode is None:
      return 0

    if self._mode == "env_id":
      privileged_info = self.env_id
    else:
      raise NotImplementedError
    return privileged_info

  def _privileged_info_space(self):
    if self._mode is None:
      return np.array([0]), np.array([1]), np.int

    if self._mode == "env_id":
      low, high, dtype = self._env_id_space()
    else:
      raise NotImplementedError
    return low, high, dtype

  @classmethod
  def instruction_wrapper(cls):
    return privileged.DummyInstructionWrapper

  @property
  def observation_space(self):
    env_id_low, env_id_high, dtype = self._env_id_space()
    privileged_info_low, privileged_info_high, dtype = self._privileged_info_space()
    data = {
        "observation": self._observation_space(),
        "env_id": gym.spaces.Box(env_id_low, env_id_high, dtype=dtype),
        "privileged_info": gym.spaces.Box(privileged_info_low, privileged_info_high, dtype=dtype),
    }
    return gym.spaces.Dict(data)


class MiniWorldConstruction(PrivilegedMiniWorldEnv, meta_exploration.MetaExplorationEnv):
  """Wrapper around the gym-miniworld Maze conforming to the MetaExplorationEnv
  interface.
  """
  action_cls = miniworld.MiniWorldEnv.Actions

  def __init__(self, env_id, wrapper):
    super().__init__(env_id, wrapper)
    self._base_env = ConstructionEnv()
    self._env = TransposeImage(self._base_env)
    self.action_space = self._env.action_space

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
    return construction_instance

  def _step(self, action):
    return self._env.step(action)

  def _reset(self):
    # Don't set the seed, otherwise can cheat from initial camera angle position!
    self._env.set_row_index(self.env_id)
    return self._env.reset()

  def _observation_space(self):
    return self._env.observation_space

  @classmethod
  def env_ids(cls):
    return list(range(3)), list(range(3))

  def _env_id_space(self):
    low = np.array([0])
    high = np.array([3])
    dtype = np.int
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
    image.write_text(f"Privileged Info ({self._mode}): {self._privileged_info}")
    return image

  def demo_policy(self):
    return self._base_env.demo_policy()


class DemoPolicy(policy.Policy):

  def __init__(self, env, valid_rows):
    self._env = env
    row = np.random.choice(valid_rows)
    self._target = np.array([self._env._size, 0, (0.5 * row + 0.15) * self._env._size])
    self._around_corner = False

  def act(self, state, hidden_state, test=False):
    del state, hidden_state

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
    collision = (
        self._env.intersect(self._env.agent, step, self._env.agent.radius))
    hit_wall = (collision is True or collision and
                not isinstance(collision, gym_miniworld.entity.Box))
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