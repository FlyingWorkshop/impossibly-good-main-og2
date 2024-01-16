import numpy as np

from envs.privileged import PrivilegedGridEnv, Action
from envs.grid import GridEnv


class LightDarkEnv(PrivilegedGridEnv):
    _goal = np.array((2, 2))
    _height = _width = 5
    _dark_cols = _width - 1
    _ids = np.random.RandomState(seed=0).permutation(_dark_cols * _height)

    def __init__(self, env_id=0, wrapper=None, max_steps=20, mode=None):
        super().__init__(env_id, wrapper, max_steps, mode=mode)
        self.obs_len = 2

    def _observation_space(self):
        low = np.array((0, 0))
        high = np.array((self._width, self._height))
        dtype = int
        return low, high, dtype

    def _compute_optimal_action(self, pos):
        if np.array_equal(pos, self._goal):
            optimal_action = Action.end_episode
        elif pos[0] < self._goal[0]:
            optimal_action = Action.right
        elif pos[0] > self._goal[0]:
            optimal_action = Action.left
        elif pos[1] < self._goal[1]:
            optimal_action = Action.up
        else:
            optimal_action = Action.down
        return optimal_action

    def _gen_obs(self):
        observed_pos = self.agent_pos.copy()
        if not self._in_light():
            observed_pos = np.array([0, 0])
        obs = super()._process_obs(observed_pos)
        return obs

    def _in_light(self):
        in_light = (
            self.agent_pos[0] >= self._light_x and
            self.agent_pos[0] < self._light_x + self._light_width and
            self.agent_pos[1] >= self._light_y and
            self.agent_pos[1] < self._light_y + self._light_height
        )
        return in_light

    def _env_id_space(self):
        low = np.array([0])
        high = np.array([self._dark_cols * self.height])
        dtype = int
        return low, high, dtype

    @classmethod
    def env_ids(cls):
        # remove the ID at the goal cell
        ids = [id for id in cls._ids if not np.array_equal(cls._goal, np.divmod(id, cls._height))]
        train_ids, test_ids = ids, ids
        return train_ids, test_ids

    def _place_objects(self):
        self._agent_pos = np.array(divmod(self.env_id, self.height))
        self._privileged_info = self._gen_privileged_info()

        self._light_x = self._width - 1
        self._light_y = 0
        self._light_width = 1
        self._light_height = self._height

    def render(self, mode="human"):
        image = super().render(mode=mode)
        # if self._obs is not None:
        #     image.draw_rectangle(self._obs['observation'], 0.8, "blue")
        for x in range(self._light_x, self._light_x + self._light_width):
            for y in range(self._light_y, self._light_y + self._light_height):
                image.draw_rectangle(np.array((x, y)), 0.5, "yellow")
        image.draw_rectangle(self._goal, 0.5, "green")
        return image
    
    def reset(self, seed=None):
        # we do this render so that we capture the last timestep in episodes
        if hasattr(self, "_agent_pos"):
            self.last_render = self.render()

        if seed is not None:
            # seed is only specified when `make_env` is called in train.py
            self._our_seed = seed
        else:
            # reset is called w/o seed in `collect_experiences`, so we should increment by number of procs
            if self._steps > 0:
                self._our_seed += self._num_procs

        # create env_id before placing objects in `super()._reset`
        random = np.random.RandomState(self._our_seed)
        ids, _ = self.env_ids()
        env_id = ids[random.randint(len(ids))]
        self._env_id = env_id
        obs = super().reset()

        # rendering
        self.last_reward = None
        self.last_action = None

        return obs, {}

