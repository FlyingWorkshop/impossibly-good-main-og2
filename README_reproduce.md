# Steps to Reproduce

## Setup
Python 3.8.10

1. Clone the original [repo](https://github.com/aaronwalsman/impossibly-good/).
2. In the new folder, create a `virtualenv` with:
```
virtualenv venv
source venv/bin/activate
```
3. Install `ltron` with `pip install ltron`
4. Build `ltron_torch` from source. You can find the folder in this [repo](https://github.com/aaronwalsman/ltron-torch-eccv22).
5. Install the remaining requirements using `pip install -r requirements.txt`. It is important that you install these dependencies after steps 3 and 4 to ensure that you have the right `gym` version.
6. Next, you will have to modify some library code to make it compatible with Python 3.8.10:
    *   Remove the typing subscripts in `/venv/lib/python3.8/site-packages/gym_minigrid/minigrid.py`
        *  line 86: change `class MissionSpace(spaces.Space[str]):` to `class MissionSpace(spaces.Space):`
        * line 104: change `seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None` to `seed: Optional = None`
7. Next, you will have to modify some of the ELF codebase.
    * In `algos/distill.py`, make it so that the `cleanup` method on line 799 is just `pass`.
    * In `envs/zoo.py` make sure that both `reset` methods `obs, {}` and that both `step` methods return 5 arguments (e.g., `obs, reward, term, term, info`)
    * Replace the observation space in `MatchingColorEnv` like:
    ```python
    # modify the obsevation space with the extra 'observed_color' variable
    # self.observation_space['observed_color'] = Discrete(len(COLOR_TO_IDX))
    # self.observation_space['expert'] = self.action_space
    # self.observation_space['step'] = Discrete(self.max_steps)

    self.observation_space = Dict({
        "direction": self.observation_space["direction"],
        "image": self.observation_space["image"],
        "mission": self.observation_space["mission"],
        "observed_color": Discrete(len(COLOR_TO_IDX)),
        "expert": self.action_space,
        "step": Discrete(self.max_steps)
    })
    ```

## Run experiments

```bash
python scripts/train.py --algo='fe' --env='ImpossiblyGood-ExampleOne-5x5-v0'
python scripts/train.py --algo='fe' --env='ImpossiblyGood-SingleBranch-v0'
python scripts/train.py --algo='fe' --env='ImpossiblyGood-EarlyExplore3-v0'
python scripts/train.py --algo='fe' --env='ImpossiblyGood-ExampleFourHard-9x9-v0'
```

