import enum
import numpy as np
import functools
from . import policy


ONION_POS = [3, 1]
ONION_DIR = 0
TOMATO_POS = [3, 3]
TOMATO_DIR = 1

# TODO: these can be removed
DISH_POS = [1, 2]
DISH_DIR = 3
POT_POS = [3, 2]
POT_DIR = 2
STATION_POS = [3, 3]
STATION_DIR = 1

ONION_POS_FOR_P1 = [6, 2]
ONION_DIR_FOR_P1 = 0
TOMATO_POS_FOR_P1 = [5, 2]
TOMATO_DIR_FOR_P1 = 0
POT_POS_FOR_P1 = [5, 2]
POT_DIR_FOR_P1 = 3
DISH_POS_FOR_P1 = [5, 3]
DISH_DIR_FOR_P1 = 3
STATION_POS_FOR_P1 = [5, 3]
STATION_DIR_FOR_P1 = 1


class Action(enum.IntEnum):
    north = 0
    south = 1
    east = 2
    west = 3
    stay = 4
    interact = 5
    end_episode = 6


class P1_DemoPolicy(policy.Policy):
    def __init__(self, env):
        self._env = env
        self._filled_ingredient = False

    def reset(self):
        self._filled_ingredient = False

    def act(self, state, hidden_state, test):
        obj = self._env._p2_policy.get_type()
        return self.fill_and_serve_policy(self._env._last_obs, obj), None

    def fill_and_serve_policy(self, obs, obj):
        pos = obs['both_agent_obs'][0][-2:]
        direction = obs['both_agent_obs'][0][:4]

        cooking = bool(obs['both_agent_obs'][0][25])
        holding_plate = obs['both_agent_obs'][0][4:8][2]
        soup_ready = bool(obs['both_agent_obs'][0][26])
        holding_soup = obs['both_agent_obs'][0][4:8][1]

        # CHECK FOR ENOUGH INGREDIENTS
        if obj == 0:    # 0 corresponds to onion
            enough_ingredients = obs['both_agent_obs'][0][27] == 3
        else:
            enough_ingredients = obs['both_agent_obs'][0][28] == 3

        if holding_soup:
            self._filled_ingredient = False
            return go_to_and_face(
                pos, direction, STATION_POS_FOR_P1, STATION_DIR_FOR_P1, interact=True)
        elif soup_ready and holding_plate:
            return go_to_and_face(
                pos, direction, POT_POS_FOR_P1, POT_DIR_FOR_P1, interact=True)
        elif holding_plate:
            return go_to_and_face(
                pos, direction, POT_POS_FOR_P1, POT_DIR_FOR_P1, interact=True)
        elif cooking or soup_ready:
            return go_to_and_face(
                pos, direction, DISH_POS_FOR_P1, DISH_DIR_FOR_P1, interact=True)
        elif enough_ingredients:
            # Start cooking
            return Action.interact
        elif not self._filled_ingredient:
            if obj == 0:
                action, done = self.fill_onion_policy(obs)
                if done:
                    self._filled_ingredient = True
                return action
            else:
                action, done = self.fill_tomato_policy(obs)
                if done:
                    self._filled_ingredient = True
                return action
        return Action.stay

    def fill_onion_policy(self, obs):
        pos = obs['both_agent_obs'][0][-2:]
        direction = obs['both_agent_obs'][0][:4]
        holding_onion = obs['both_agent_obs'][0][4:8][0]
        at_pot = np.array_equal(pos, POT_POS_FOR_P1) and direction[POT_DIR_FOR_P1] == 1
        if not holding_onion:
            return go_to_and_face(
                pos, direction, ONION_POS_FOR_P1, ONION_DIR_FOR_P1, interact=True), False
        else:
            if at_pot and holding_onion:
                return Action.interact, True
            return go_to_and_face(
                pos, direction, POT_POS_FOR_P1, POT_DIR_FOR_P1, interact=False), False

    def fill_tomato_policy(self, obs):
        pos = obs['both_agent_obs'][0][-2:]
        direction = obs['both_agent_obs'][0][:4]
        holding_tomato = obs['both_agent_obs'][0][4:8][-1]
        at_pot = np.array_equal(pos, POT_POS_FOR_P1) and direction[POT_DIR_FOR_P1] == 1
        if not holding_tomato:
            return go_to_and_face(
                pos, direction, TOMATO_POS_FOR_P1, TOMATO_DIR_FOR_P1, interact=True), False
        else:
            if at_pot and holding_tomato:
                return Action.interact, True
            return go_to_and_face(
                pos, direction, POT_POS_FOR_P1, POT_DIR_FOR_P1, interact=False), False


def go_to_and_face(pos, dir, target_pos, target_dir, interact=False):
    if pos[1] > target_pos[1]: return Action.north
    if pos[1] < target_pos[1]: return Action.south
    if pos[0] < target_pos[0]: return Action.east
    if pos[0] > target_pos[0]: return Action.west
    if dir[target_dir] != 1: return target_dir
    if interact: return Action.interact
    return Action.stay


def fill_onion_policy(obs, ingredients_filled_by_this_agent=0, num_ingredients=1):
    pos = obs['both_agent_obs'][1][-2:]
    direction = obs['both_agent_obs'][1][:4]

    holding_onion = obs['both_agent_obs'][1][4:8][0]
    # enough_onions = obs['both_agent_obs'][1][27] == num_ingredients
    enough_onions = ingredients_filled_by_this_agent >= num_ingredients
    cooking = bool(obs['both_agent_obs'][1][25])
    soup_ready = bool(obs['both_agent_obs'][1][26])
    holding_soup = bool(obs['both_agent_obs'][0][4:8][1])
    at_pot = np.array_equal(pos, POT_POS) and direction[POT_DIR] == 1

    if soup_ready or cooking or enough_onions:
        return Action.stay, holding_soup, ingredients_filled_by_this_agent
    elif not holding_onion:
        return go_to_and_face(
            pos, direction, ONION_POS, ONION_DIR, interact=True), False, ingredients_filled_by_this_agent
    else:
        if at_pot and holding_onion:
            ingredients_filled_by_this_agent += 1
        return go_to_and_face(
            pos, direction, POT_POS, POT_DIR, interact=True), False, ingredients_filled_by_this_agent


def fill_tomato_policy(obs, ingredients_filled_by_this_agent=0, num_ingredients=1):
    pos = obs['both_agent_obs'][1][-2:]
    direction = obs['both_agent_obs'][1][:4]    

    holding_tomato = obs['both_agent_obs'][1][4:8][-1]
    # enough_tomatoes = obs['both_agent_obs'][1][28] == num_ingredients
    enough_tomatoes = ingredients_filled_by_this_agent >= num_ingredients
    cooking = bool(obs['both_agent_obs'][1][25])
    soup_ready = bool(obs['both_agent_obs'][1][26])
    holding_soup = bool(obs['both_agent_obs'][0][4:8][1])
    at_pot = np.array_equal(pos, POT_POS) and direction[POT_DIR] == 1

    if soup_ready or cooking or enough_tomatoes:
        return Action.stay, holding_soup, ingredients_filled_by_this_agent
    elif not holding_tomato:
        return go_to_and_face(
            pos, direction, TOMATO_POS, TOMATO_DIR, interact=True), False, ingredients_filled_by_this_agent
    else:
        if at_pot and holding_tomato:
            ingredients_filled_by_this_agent += 1
        return go_to_and_face(
            pos, direction, POT_POS, POT_DIR, interact=True), False, ingredients_filled_by_this_agent


def fill_and_serve_policy(obs, obj, num_ingredients=1):
    pos = obs['both_agent_obs'][1][-2:]
    direction = obs['both_agent_obs'][1][:4]

    cooking = bool(obs['both_agent_obs'][1][25])
    holding_plate = obs['both_agent_obs'][1][4:8][2]
    soup_ready = bool(obs['both_agent_obs'][1][26])
    holding_soup = obs['both_agent_obs'][1][4:8][1]

    if holding_soup:
        at_station = (
            pos[0] == STATION_POS[0] and
            pos[1] == STATION_POS[1] and
            direction[STATION_DIR] == 1
        )
        if at_station:
            return Action.interact, True
        return go_to_and_face(
            pos, direction, STATION_POS, STATION_DIR, interact=False), False
    elif soup_ready and holding_plate:
        return go_to_and_face(
            pos, direction, POT_POS, POT_DIR, interact=True), False
    elif holding_plate:
        return go_to_and_face(
            pos, direction, POT_POS, POT_DIR, interact=False), False
    elif cooking or soup_ready:
        return go_to_and_face(
            pos, direction, DISH_POS, DISH_DIR, interact=True), False
    else:
        if obj == 'onion':
            return fill_onion_policy(obs, num_ingredients)[0], False
        else:
            return fill_tomato_policy(obs, num_ingredients)[0], False


def fill_onion_and_serve_policy(obs, num_ingredients=1):
    return fill_and_serve_policy(obs, 'onion', num_ingredients)


def fill_tomato_and_serve_policy(obs, num_ingredients=1):
    return fill_and_serve_policy(obs, 'tomato', num_ingredients)


class P2_Policy(object):
    POLICIES = [
        fill_onion_policy,
        fill_tomato_policy,
        # fill_onion_and_serve_policy,
        # fill_tomato_and_serve_policy,
    ]

    def __init__(self, np_random=None):
        self._np_random = np_random
        if self._np_random is None:
            print("WARNING: np_random is None, using np.random.RandomState()")
            self._np_random = np.random.RandomState()

        self._policy_idx = self._np_random.randint(len(self.POLICIES))
        self._num_ingredients = 2
        self._ingredients_filled = 0
        self._policy = functools.partial(
            self.POLICIES[self._policy_idx],
            num_ingredients=self._num_ingredients)

    def act(self, obs):
        action, done, self._ingredients_filled = self._policy(obs, ingredients_filled_by_this_agent=self._ingredients_filled)
        if done:
            self._policy_idx = self._np_random.randint(len(self.POLICIES))
            self._policy = functools.partial(
                self.POLICIES[self._policy_idx],
                num_ingredients=self._num_ingredients)
            self._ingredients_filled = 0
        return action

    def get_type(self):
        return self._policy_idx