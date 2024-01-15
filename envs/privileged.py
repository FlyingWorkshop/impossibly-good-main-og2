import abc
import enum

import gym
import numpy as np

from envs import grid
from . import meta_exploration
from . import policy


class Action(enum.IntEnum):
    left = 0
    up = 1
    right = 2
    down = 3
    end_episode = 4


class DummyInstructionWrapper(meta_exploration.InstructionWrapper):
    def _instruction_observation_space(self):
        return gym.spaces.Box(np.array([0]), np.array([1]), dtype=np.int)

    def _reward(self, instruction_state, action, original_reward):
        return original_reward, False

    def _generate_instructions(self, test=False):
        del test
        goal = np.array((0))
        return goal

class PrivilegedGridEnv(grid.GridEnv):
    _gym_disable_underscore_compat = True

    _blocked = []
    _failure = None
    _goal = None
    _privileged_info_modes = ("env_id", "dynamic-demo", "static-demo", "env-id-dynamic-demo", "optimal-act", "init-pos", "curr-pos", None)
    _refreshing_modes = ("dynamic-demo", "optimal-act", "env-id-dynamic-demo", "curr-pos")
    action_cls = Action

    def __init__(self, env_id, wrapper, max_steps=20, mode=None):
        assert mode in self._privileged_info_modes, f"{mode} is not a valid privileged info mode!"

        super().__init__(env_id, wrapper, max_steps, width=self._width, height=self._height)
        self._path = None
        self._obs = None
        self._refresh = mode in self._refreshing_modes
        self._mode = mode
        self._privileged_info = None  # IMPORTANT: make sure to call self._gen_privileged_info() in concrete subclasses to avoid rendering bug

        # added for ELF
        self._last_reward = None
        self._last_action = None
        self._rng = None
        self._our_seed = None
        self._num_procs = None
        self.max_steps = self._max_steps

    def _gen_privileged_info(self):
        if self._mode is None:
            return 0

        if self._mode == "env_id":
            privileged_info = self.env_id
        elif "demo" in self._mode:
            privileged_info = self._compute_demo()
        elif self._mode == "init-pos" or self._mode == "curr-pos":
            privileged_info = self.agent_pos
        elif self._mode == "optimal-act":
            optimal_action = self._compute_optimal_action(self.agent_pos)
            if optimal_action is None:
                return
            privileged_info = optimal_action.value
        return privileged_info

    def _privileged_info_space(self):
        if self._mode is None:
            return np.array([0]), np.array([1]), np.int

        if self._mode == "env_id":
            low, high, dtype = self._env_id_space()
        elif "demo" in self._mode:
            low, high, dtype = self._demo_space()
        elif self._mode == "init-pos" or self._mode == "curr-pos":
            low, high, dtype = self._observation_space()
        elif self._mode == "optimal-act":
            low, high, dtype = self._optimal_action_space()
        return low, high, dtype

    def _optimal_action_space(self):
        low = np.array([0])
        high = np.array([len(self.action_cls)])
        dtype = np.int
        return low, high, dtype

    def _demo_space(self):
        observation_high = self._observation_space()[1]
        low = np.zeros((self._max_steps, len(observation_high))).ravel()
        high = np.tile(observation_high, (self._max_steps, 1)).ravel()
        dtype = np.int
        return low, high, dtype

    def _compute_demo_actions(self, pos):
        actions = []
        while (optimal_action := self._compute_optimal_action(pos)) is not None:
            pos = self._calc_next_pos(pos, optimal_action)
            actions.append(optimal_action)
        return actions

    def _compute_demo(self):
        self._path = self._compute_path()
        if self._mode == "env-id-dynamic-demo":
            observed_path = [np.concatenate((pos, [self.env_id])) for pos in self._path]
        else:
            observed_path = self._observe_path(self._path)
        pad_len = self._max_steps - len(observed_path)
        demo = np.pad(observed_path, ((0, pad_len), (0, 0)), mode='edge')
        return demo.ravel()

    def _compute_path(self):
        path = [self.agent_pos]
        while (optimal_action := self._compute_optimal_action(path[-1])) is not None:
            path.append(self._calc_next_pos(path[-1], optimal_action))
        return path

    def _observe_path(self, path):
        """Monkey patch agent position to observe path"""
        orig_agent_pos = self.agent_pos
        observed_path = []
        for pos in path:
            self._agent_pos = pos
            observed_path.append(self._gen_obs())
        self._agent_pos = orig_agent_pos
        return observed_path

    @abc.abstractmethod
    def _compute_optimal_action(self, pos: np.ndarray, dir: int):
        raise NotImplementedError

    def _calc_next_pos(self, curr_pos: np.ndarray, action: Action):
        next_pos = np.copy(curr_pos)
        if action == Action.left:
            next_pos[0] -= 1
        elif action == Action.up:
            next_pos[1] += 1
        elif action == Action.right:
            next_pos[0] += 1
        elif action == Action.down:
            next_pos[1] -= 1
        elif action == Action.end_episode:
            pass
        else:
            assert False, "Invalid action!"

        next_pos = np.clip(next_pos, [0, 0], [self.width - 1, self.height - 1])

        if tuple(next_pos) in self._blocked:
            next_pos = curr_pos

        return next_pos

    def _step(self, action: Action):
        self._steps += 1
        self._agent_pos = self._calc_next_pos(self.agent_pos, action)
        self._obs = self._gen_obs()
        self._history.append(np.array(self._agent_pos))

        if self._refresh:
            self._privileged_info = self._gen_privileged_info()

        if np.array_equal(self.agent_pos, self._goal):
            reward = 1
            done = True
        elif np.array_equal(self.agent_pos, self._failure):
            reward = -1
            done = True
        else:
            reward = -.1
            done = self._steps == self._max_steps

        # if action == Action.end_episode:
        #     reward -= self.steps_remaining * 0.1  # penalize ending the episode
        #     done = True

        # added for ELF
        self.last_action = Action(action)
        self.last_reward = reward
        self.last_render = self.render()

        return self._obs, reward, done, {}

    @classmethod
    def instruction_wrapper(cls):
        return meta_exploration.DummyInstructionWrapper

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.action_cls))

    @property
    def observation_space(self):
        observation_low, observation_high, dtype = self._observation_space()
        # env_id_low, env_id_high, _ = self._env_id_space()
        # privileged_info_low, privileged_info_high, dtype = self._privileged_info_space()
        data = {
            "observation": gym.spaces.Box(observation_low, observation_high, dtype=int),
            # "env_id": gym.spaces.Box(env_id_low, env_id_high, dtype=dtype),
            # "privileged_info": gym.spaces.Box(privileged_info_low, privileged_info_high, dtype=dtype),
            "expert": gym.spaces.Box(0, len(self.action_cls), shape=(), dtype=self.action_cls),
            "step": gym.spaces.Box(0, self.max_steps, shape=(), dtype=int)
        }
        return gym.spaces.Dict(data)

    def reset(self):
        return super()._reset()
        # # we do this render so that we capture the last timestep in episodes
        # if hasattr(self, "_agent_pos"):
        #     self.last_render = self.render()

        # if seed is not None:
        #     # seed is only specified when `make_env` is called in train.py
        #     self._seed = seed
        # else:
        #     # reset is called w/o seed in `collect_experiences`, so we should increment by number of procs
        #     self._seed += self._num_procs

        # # self._rng = np.random.RandomState(self._seed)

        # # create env_id before placing objects in `super()._reset`
        # self._env_id = 0
        # # self._env_id = self.create_env_id(self._rng.randint(1e5))
        # obs = super()._reset()

        # # rendering
        # self.last_reward = None
        # self.last_action = None

        # return obs, {}

    def step(self, action):
        obs, reward, done, info = self._step(action)
        return obs, reward, done, done, info

    def _process_obs(self, obs):
        optimal_action = self._compute_optimal_action(self.agent_pos)
        obs = {"observation": obs, "expert": optimal_action, "step": self._steps}
        return obs

    def render(self, mode="human"):
        image = super().render(mode)
        optimal_action = self._compute_optimal_action(self.agent_pos)
        image.write_text("Expert: {}".format(optimal_action.__repr__()))  # optimal next action
        image.write_text("Action: {}".format(self.last_action.__repr__()))  # last action
        image.write_text("Reward: {}".format(self.last_reward))  # last reward
        image.write_text("Timestep: {}".format(self._steps))  # current timestep
        if optimal_action is not None:
            pos = self._calc_next_pos(self.agent_pos, optimal_action)
            image.draw_rectangle(pos, 0.1, "indigo")
        return image

    def demo_policy(self):
        return DemoPolicy(self, self._compute_demo_actions)


class DemoPolicy(policy.Policy):

  def __init__(self, env, demo_actions):
    self._env = env
    self._demo_actions = demo_actions
    self._action_seq = None

  def act(self, state, hidden_state, test=False):
    del hidden_state, test

    if self._action_seq is None:
      self._action_seq = iter(self._demo_actions(self._env.agent_pos))

    try:
      next_action = next(self._action_seq)
      return next_action, None
    except StopIteration:
      # Reset policy
      self._action_seq = iter(self._demo_actions(state))
      next_action = next(self._action_seq)
      return next_action, None