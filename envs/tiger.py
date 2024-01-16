import numpy as np

from envs.privileged import PrivilegedGridEnv, Action


class TigerDoorEnv(PrivilegedGridEnv):
    _blocked = [
        (1, 0), 
        (1, 1), 
        (1, 3),
        (2, 0),
        (3, 0), 
        (3, 2),
        (4, 0), 
        (4, 2),
    ]

    _initial_position = np.array((0, 0))
    _top = np.array((4, 1))
    _bot = np.array((4, 3))
    _pink = np.array((0, 3))  # moving here reveals if goal is top or bottom cell

    _width = 5
    _height = 4

    _optimal_action_to_top = {
        (0, 0): Action.up,
        (0, 1): Action.up,
        (0, 2): Action.right,
        (0, 3): Action.down,
        (1, 2): Action.right,
        (2, 1): Action.right,
        (2, 2): Action.down,
        (2, 3): Action.down,
        (3, 1): Action.right,
        (3, 3): Action.left,
        (4, 1): Action.end_episode,
        (4, 3): Action.left
    }

    _optimal_action_to_bot = {
        (0, 0): Action.up,
        (0, 1): Action.up,
        (0, 2): Action.right,
        (0, 3): Action.down,
        (1, 2): Action.right,
        (2, 1): Action.up,
        (2, 2): Action.up,
        (2, 3): Action.right,
        (3, 1): Action.left,
        (3, 3): Action.right,
        (4, 1): Action.left,
        (4, 3): Action.end_episode
    }

    def __init__(self, env_id=0, wrapper=None, max_steps=20, mode=None):
        super().__init__(env_id, wrapper, max_steps, mode)
        self.obs_len = 3

    def _observation_space(self):
        env_id_low, env_id_high, dtype = self._env_id_space()
        low = np.concatenate(([0, 0], env_id_low))
        high = np.concatenate(([self.width, self.height], env_id_high + 1))  # +1 since observered env_id could be unknown
        return low, high, dtype

    def _env_id_space(self):
        low = np.array([0])
        high = np.array([2])
        dtype = int
        return low, high, dtype

    def _should_move_to_top(self) -> bool:
        return bool(self.env_id)

    def _compute_optimal_action(self, pos):
        if self._should_move_to_top():
            optimal_action = self._optimal_action_to_top[tuple(pos)]
        else:
            optimal_action = self._optimal_action_to_bot[tuple(pos)]
        return optimal_action

    def _gen_obs(self):
        if np.array_equal(self.agent_pos, self._pink):
            observed_id = self.env_id
        else:
            observed_id = self._env_id_space()[1][0]
        obs = np.concatenate((self.agent_pos, [observed_id]))
        obs = super()._process_obs(obs)
        return obs

    @classmethod
    def env_ids(cls):
        train_ids = [0, 1]
        test_ids = [0, 1]
        return train_ids, test_ids

    def _place_goal(self, is_top: bool):
        if is_top:
            self._goal = self._top
            self._failure = self._bot
        else:
            self._goal = self._bot
            self._failure = self._top

    def _place_objects(self):
        self._agent_pos = self._initial_position
        self._place_goal(bool(self.env_id))
        self._privileged_info = self._gen_privileged_info()

    def render(self, mode="human"):
        image = super().render(mode=mode)
        for cell in self._blocked:
            image.draw_rectangle(np.array(cell), 1.0, "grey")
        image.draw_rectangle(self._goal, 0.5, "green")
        image.draw_rectangle(self._failure, 0.5, "blue")
        image.draw_rectangle(self._pink, 0.5, "pink")
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
