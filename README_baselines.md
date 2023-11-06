# Baselines (Training):

For debugging, just add `procs=1`

## ELF Envs
TODO: make sure that these baselines can handle the render flag
```bash
python scripts/train.py --algo='fe' --env='ImpossiblyGood-ExampleOne-5x5-v0'
python scripts/train.py --algo='fe' --env='ImpossiblyGood-SingleBranch-v0'
python scripts/train.py --algo='fe' --env='ImpossiblyGood-EarlyExplore3-v0'
python scripts/train.py --algo='fe' --env='ImpossiblyGood-ExampleFourHard-9x9-v0'
```

## DREAM Envs
```bash
python scripts/train.py --algo='fe' --env='ImpossiblyGood-TigerDoor-v0' --render
python scripts/train.py --algo='fe' --env='ImpossiblyGood-LightDark-v0' --render
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python scripts/train.py --algo='fe' --env='ImpossiblyGood-Construction-v0' --render
```

# Evaluation

## ELF Envs

## DREAM Envs
```bash
python scripts/evaluate.py --env='ImpossiblyGood-TigerDoor-v0' --model='storage-v1/ImpossiblyGood-TigerDoor-v0_fea_seed1_23-10-24-06-52-08' --procs=1
```

# Testing on Workstations

```bash
ssh loganmb@sc.stanford.edu
srun --account=iris -p iris-interactive --mem=8GB --gres=gpu:1 --pty --exclude=iris5,iris6,iris7 bash
cd /iris/u/loganmb/impossibly-good-main
source venv/bin/activate
```