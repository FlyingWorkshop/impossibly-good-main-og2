# Baselines (Training):

For debugging, just add `procs=1`

`python scripts/train.py --algo='fe' --env='ImpossiblyGood-NonstationaryMap-v0' --render --eval-argmax --eval-episodes=10
--procs=1
`

## DREAM Envs
```bash
python scripts/train.py --algo='fe' --env='ImpossiblyGood-TigerDoor-v0' --render --eval-argmax --eval-episodes=10
python scripts/train.py --algo='fe' --env='ImpossiblyGood-LightDark-v0' --render --eval-argmax --eval-episodes=10
python scripts/train.py --algo='fe' --env='ImpossiblyGood-NonstationaryMap-v0' --render --eval-argmax --eval-episodes=10
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python scripts/train.py --algo='fe' --env='ImpossiblyGood-Construction-v0' --render --eval-argmax --eval-episodes=10
```

## ELF Envs
```bash
python scripts/train.py --algo='fe' --env='ImpossiblyGood-ExampleOne-5x5-v0' --render --eval-episodes=10
python scripts/train.py --algo='fe' --env='ImpossiblyGood-SingleBranch-v0' --render --eval-episodes=10
python scripts/train.py --algo='fe' --env='ImpossiblyGood-EarlyExplore3-v0' --render --eval-episodes=10
python scripts/train.py --algo='fe' --env='ImpossiblyGood-ExampleFourHard-9x9-v0' --render --eval-episodes=10
```

# Testing on Workstations

Note that some experiments will fail without at least 16GB.
```bash
ssh $USER@sc.stanford.edu
srun --account=iris -p iris-interactive --mem=16GB --gres=gpu:1 --pty --exclude=iris5,iris6,iris7 bash
cd /iris/u/$USER/impossibly-good-main
source venv/bin/activate
```