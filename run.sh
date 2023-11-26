#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=iris5,iris6,iris7
# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="tiger"
#SBATCH --output=out/%j.out
#SBATCH --time=72:0:0

cd /iris/u/loganmb/impossibly-good-main
source venv/bin/activate

## ELF Envs
# python scripts/train.py --algo='fe' --env='ImpossiblyGood-ExampleOne-5x5-v0' --render --eval-episodes=10
# python scripts/train.py --algo='fe' --env='ImpossiblyGood-SingleBranch-v0' --render --eval-episodes=10
# python scripts/train.py --algo='fe' --env='ImpossiblyGood-EarlyExplore3-v0' --render --eval-episodes=10
# python scripts/train.py --algo='fe' --env='ImpossiblyGood-ExampleFourHard-9x9-v0' --render --eval-episodes=10

## DREAM Envs
# python scripts/train.py --algo='fe' --env='ImpossiblyGood-TigerDoor-v0' --render --eval-argmax --eval-episodes=10
# python scripts/train.py --algo='fe' --env='ImpossiblyGood-LightDark-v0' --render --eval-argmax --eval-episodes=10
# python scripts/train.py --algo='fe' --env='ImpossiblyGood-NonstationaryMap-v0' --render --eval-argmax --eval-episodes=10
# xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python scripts/train.py --algo='fe' --env='ImpossiblyGood-Construction-v0' --render --eval-argmax --eval-episodes=10