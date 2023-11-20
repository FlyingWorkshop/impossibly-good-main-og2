import itertools

import gym
import numpy as np

from envs import grid
from envs import meta_exploration
from envs.grid import Action


class InstructionWrapper(meta_exploration.InstructionWrapper):
  """Instruction wrapper for CityGridEnv.

  Provides instructions (goal locations) and their corresponding rewards.

  Reward function for a given goal is:
      R(s, a) = -0.1 if s != goal
              = 1    otherwise
  """

  def _instruction_observation_space(self):
    return gym.spaces.Box(
        np.array([0, 0]), np.array([self.width, self.height]), dtype=int)

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
    self.env._goal = goal
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
  
  def reset(self, seed=None):
    return super().reset(seed=seed)


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
    dtype = int
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
  """Wrapper to create non-stationary GridEnv."""
  _gym_disable_underscore_compat = True

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

    # added for ELF
    self.last_action = None
    self.last_reward = None
    self._rng = None
    self.max_steps = self._max_steps
    self.obs_len = 4

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

  def reset(self, seed=None):
      # we do this render so that we capture the last timestep in episodes
      if hasattr(self, "_agent_pos"):
          self.last_render = self.render()

      # create new env_id before calling super reset (which places objects)
      if seed is not None:
          self._rng = np.random.RandomState(seed)
      assert self._rng is not None
      self._env_id = self.create_env_id(self._rng.randint(1e5))
      
      obs = self._reset()

      # rendering
      self.last_reward = None
      self.last_action = None

      return obs, {}

  def step(self, action):
    obs, reward, done, info = self._step(action)
    self.last_action = Action(action)
    self.last_reward = reward
    self.last_render = self.render()
    return obs, reward, done, done, info

  @classmethod
  def instruction_wrapper(cls):
    return NonstationaryInstructionWrapper

  def _privileged_info_space(self):
    if self._mode is None:
      return np.array([0]), np.array([1]), int

    if self._mode == "env_id":
      low, high, dtype = self._env_id_space()
    elif self._mode == "all-env-ids":
      low, high, dtype = self._env_id_space()
      low = low.repeat(self._max_steps)
      high = high.repeat(self._max_steps)
    return low, high, dtype

  # NOTE: not really used
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
    optimal_action = self._compute_optimal_action(self.agent_pos)
    image.write_text("Expert: {}".format(optimal_action.__repr__()))  # optimal next action
    image.write_text("Action: {}".format(self.last_action.__repr__()))  # last action
    image.write_text("Reward: {}".format(self.last_reward))  # last reward
    image.write_text("Timestep: {}".format(self._steps))  # current timestep
    if optimal_action in (Action.left, Action.right, Action.down, Action.up):
      next_pos = np.copy(self.agent_pos)
      if optimal_action == Action.left:
          next_pos[0] -= 1
      elif optimal_action == Action.up:
          next_pos[1] += 1
      elif optimal_action == Action.right:
          next_pos[0] += 1
      elif optimal_action == Action.down:
          next_pos[1] -= 1
      image.draw_rectangle(next_pos, 0.1, "indigo")
    return image

  def _get_goal_bus(self):
    distances = np.linalg.norm(self._destinations - self._goal, ord=1, axis=1)
    i = np.argwhere(distances == 1)[0][0]
    bus = self.get(self._destinations[i])._destination
    for bus_source, _ in self._bus_sources:
      bus = self.get(bus_source)
      if np.linalg.norm(bus._destination - self._goal, ord=1) == 1:
        return bus, bus_source

  @staticmethod
  def _get_walking_directions(start, stop):
    if start[0] < stop[0]:
      return Action.right
    elif start[0] > stop[0]:
      return Action.left
    elif start[1] < stop[1]:
      return Action.up
    elif start[1] > stop[1]:
      return Action.down
    else:
      return None

  def _get_nearest_bus_by_dest(self, pos):
    nearest_bus = None
    nearest_dist = np.inf
    nearest_bus_source = None
    for bus_source, _ in self._bus_sources:
      bus = self.get(bus_source)
      dist = self._calc_dist(bus._destination, pos)
      if dist < nearest_dist:
        nearest_bus = bus
        nearest_dist = dist
        nearest_bus_source = None
    return nearest_bus, nearest_bus_source

  @staticmethod     
  def _calc_dist(a, b):
    return np.linalg.norm(a - b, ord=1)

  def _compute_naive_optimal_action_old(self, pos: np.ndarray):
    """
    Compute the naive (non-future-aware) optimal action

    Goal: get to goal as quickly as possible
    Strategy:
    - the agent is either (1) near goal, (2) near center, or (3) near non-goal corner.
    (1): walk to goal
    (2): take goal bus to goal
    (3): take nearest bus to center
    """
    if np.array_equal(pos, self._goal):
      return None

    # Case (1):
    walk_time = self._calc_dist(pos, self._goal)

    # Case (2):
    goal_bus, goal_bus_source = self._get_goal_bus()
    walk_to_bus_time = self._calc_dist(pos, goal_bus_source) 
    ride_time = 1
    goal_bus_dest_to_goal_time = self._calc_dist(pos, goal_bus._destination)
    walk_then_bus_time = walk_to_bus_time + ride_time + goal_bus_dest_to_goal_time

    # Case (3):
    nearest_bus, nearest_bus_source = self._get_nearest_bus_by_dest(pos)
    bus_to_bus_time = self._calc_dist(pos, nearest_bus._destination) + ride_time + self._calc_dist(nearest_bus_source, goal_bus_source)
    bus_then_bus_time = bus_to_bus_time + ride_time + goal_bus_dest_to_goal_time

    if walk_time == min(walk_time, walk_then_bus_time, bus_then_bus_time):
      target = self._goal
    elif walk_then_bus_time <= bus_then_bus_time:
      target = goal_bus_source
    else:
      target = nearest_bus._destination

    # if already reached target, optimal action is riding the bus
    return self._get_walking_directions(pos, target) or Action.ride_bus

  def _compute_naive_optimal_action(self, pos):
    if np.array_equal(pos, self._goal):
      return None
  
    walk_time = self._calc_dist(pos, self._goal)

    temp = self._destinations - self._goal
    breakpoint()

    

  def _walk_then_wait_time(self, steps, pos, bus_source):
    """
    Returns the timestep that the agent would complete the episode at if they walked to the given bus source from
    a given position, waited for the bus to go to the goal, rode it, then walked to the goal.

    TLDR: returns the episode end time if they use the given bus
    """
    walk_time = self._calc_dist(pos, bus_source)
    ride_time = 1
    goal_bus_dest_to_goal_dist = 1  # assumes the goal is exactly 1 action away from some bus stop
    eta = steps + walk_time + ride_time

    if eta > self._max_steps:
      return np.inf

    # figure out what the interval_index will be when agent is at bus stop
    switching_times = np.cumsum(self._switch_intervals)
    for interval_index, t in enumerate(switching_times):
      if eta <= t:
        # monkeypatch env_id to check if bus will go to goal in the future
        true_env_id = self._env_id
        env_id = self._env_ids[interval_index]
        self._switch_bus_permutation(env_id)
        bus = self.get(bus_source)
        is_goal_bus = self._calc_dist(bus._destination, self._goal) == goal_bus_dest_to_goal_dist
        self._env_id = true_env_id
        if is_goal_bus:
          if interval_index == 0:
            wait_time = 0
          else:
            wait_time = min(0, switching_times[interval_index - 1] - eta)
          break
    else:
      wait_time = np.inf
    
    return walk_time + wait_time + ride_time + goal_bus_dest_to_goal_dist
  
  def _bus_then_wait_time(self, steps, pos, bus_destination):
    """
    Returns the end time if the agent takes the nearest bus to the center then takes the best bus to the goal.
    """
    walk_time = self._calc_dist(pos, bus_destination)
    ride_time = 1
    eta = steps + walk_time + ride_time

    if eta > self._max_steps:
      return np.inf

    # get env_id at t=eta
    switching_times = np.cumsum(self._switch_intervals)
    for interval_index, t in enumerate(switching_times):
      if eta <= t:
        break
    
    # monkeypatch env_id to figure out where the bus goes
    true_env_id = self._env_id
    env_id = self._env_ids[interval_index]
    self._switch_bus_permutation(env_id)
    for nearest_bus_source, _ in self._bus_sources:
      bus = self.get(nearest_bus_source)
      if np.array_equal(bus._destination, bus_destination):
        break
    self._env_id = true_env_id

    fastest_end_time = np.inf
    for bus_source, _ in self._bus_sources:
      end_time = self._walk_then_wait_time(eta, nearest_bus_source, bus_source)
      fastest_end_time = min(end_time, fastest_end_time)
    
    return fastest_end_time
  
  def _calc_next_pos(self, pos, action, step):
    next_pos = np.copy(self.agent_pos)
    if action == Action.left:
      next_pos[0] -= 1
    elif action == Action.up:
      next_pos[1] += 1
    elif action == Action.right:
      next_pos[0] += 1
    elif action == Action.down:
      next_pos[1] -= 1
    elif action == Action.ride_bus:
      bus = self.get(pos)
      pass
    else:
      assert False, "Invalid action!"

    next_pos = np.clip(next_pos, [0, 0], [self.width - 1, self.height - 1]) 
    return next_pos

  def _compute_path(self):
    path = [self.agent_pos]
    i = 0
    while (optimal_action := self._compute_optimal_action(path[-1])) is not None:
        path.append(self._calc_next_pos(path[-1], optimal_action, self._steps + i))
        i += 1
    return path
    
  def _foo(self, pos: np.ndarray):
    if np.array_equal(pos, self._goal):
      return None
        
    fastest_end_time = self._steps + self._calc_dist(pos, self._goal)
    target = self._goal
    for bus_source, _ in self._bus_sources:
      end_time = self._walk_then_wait_time(self._steps, pos, bus_source)
      if end_time < fastest_end_time:
        fastest_end_time = end_time
        target = bus_source
      
      bus_destination = self.get(bus_source)._destination
      end_time = self._bus_then_wait_time(self._steps, pos, bus_destination)
      if end_time < fastest_end_time:
        fastest_end_time = end_time
        target = bus_destination
    
    optimal_action = self._get_walking_directions(pos, target)
    if optimal_action is None:
      # TODO: detect if should be waiting or taking the bus
      waiting = False
      if waiting:
        return Action.noop
      else:
        return Action.ride_bus
    else:
      return optimal_action

  def _process_obs(self, obs):
    optimal_action = self._compute_optimal_action(self.agent_pos)
    obs = {"observation": obs, "expert": optimal_action, "step": self._steps}
    return obs

  def _gen_obs(self):
    return self._process_obs(super()._gen_obs())

  def _get_central_bus_end_times(self, steps, pos):
    results = []
    ride_time = 1
    central_bus_sources = [bus_source for bus_source, _ in self._bus_sources]
    eta_arr = steps + np.linalg.norm(pos - central_bus_sources, ord=1, axis=1) + ride_time
    switching_times = np.cumsum(self._switch_intervals)
    
    # monkeypatch env_id
    true_env_id = self._env_id
    for bus_source, eta in zip(central_bus_sources, eta_arr):
      for interval_index, t in enumerate(switching_times):
        if eta <= t:
          env_id = self._env_ids[interval_index]
          self._switch_bus_permutation(env_id)
          goes_to_goal = np.linalg.norm(self.get(bus_source)._destination - self._goal, ord=1) == 1
          if goes_to_goal:
            if interval_index:
              wait_time = min(0, switching_times[interval_index - 1] - eta)
            else:
              wait_time = 0
            end_time = eta + wait_time + 1  # +1 to walk from bus stop to goal
            break
      else:
        end_time = np.inf
      results.append(end_time)
    self._env_id = true_env_id
    return results

  def _get_outer_bus_end_times(self, steps, pos):
    results = []
    ride_time = 1
    eta_arr = steps + np.linalg.norm(pos - self._destinations, ord=1, axis=1) + ride_time
    switching_times = np.cumsum(self._switch_intervals)

    # monkeypatch env_id
    true_env_id = self._env_id
    for dest, eta in zip(self._destinations, eta_arr):
      for interval_index, t in enumerate(switching_times):
        if eta <= t:
          break
      env_id = self._env_ids[interval_index]
      self._switch_bus_permutation(env_id)
      end_time = min(self._get_central_bus_end_times(eta, self.get(dest)._destination))
      results.append(end_time)
    self._env_id = true_env_id
    return results

  def _compute_optimal_action(self, pos: np.ndarray):
    if np.array_equal(pos, self._goal):
      return None

    best_time = self._calc_dist(pos, self._goal)
    target = self._goal
    waiting = False

    central_bus_end_times = self._get_central_bus_end_times(self._steps, pos)
    
    for i, end_time in enumerate(central_bus_end_times):
      if end_time < best_time:
        best_time = end_time
        target = self._bus_sources[i][0]
        already_at_target = np.array_equal(pos, target)
        target_goes_to_goal = np.linalg.norm(self.get(target)._destination - self._goal, ord=1) == 1
        waiting = already_at_target and target_goes_to_goal
    
    outer_bus_end_times = self._get_outer_bus_end_times(self._steps, pos)
    for i, end_time in enumerate(outer_bus_end_times):
      if end_time < best_time:
        best_time = end_time
        target = self._destinations[i]
        waiting = False  # NOTE: is is never optimal to wait at an outer bus stop
    
    action = self._get_walking_directions(pos, target)
    if action is None:
      if waiting:
        return Action.noop
      else:
        return Action.ride_bus
    else:
      return action