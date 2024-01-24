#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --exclude=iris5,iris6,iris7
# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name="con"
#SBATCH --output=out/%j.out
#SBATCH --time=72:0:0

cd /iris/u/loganmb/impossibly-good-main
source venv/bin/activate

# LightDark
# xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python scripts/train.py --algo='fe' --env='ImpossiblyGood-LightDark-v0' --render --eval-argmax --eval-episodes=100 --procs=8 --eval-frequency=5000 --seed=4

# TigerDoor
# xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python scripts/train.py --algo='fe' --env='ImpossiblyGood-TigerDoor-v0' --render --eval-argmax --eval-episodes=100 --procs=8 --eval-frequency=5000 --seed=4

# Map
# xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python scripts/train.py --algo='fe' --env='ImpossiblyGood-Map-v0' --render --eval-argmax --eval-episodes=100 --procs=8 --eval-frequency=5000 --seed=4

# # NS Map
# xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python scripts/train.py --algo='fe' --env='ImpossiblyGood-NonstationaryMap-v0' --render --eval-argmax --eval-episodes=100 --procs=8 --eval-frequency=5000 --seed=4

# Construction
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python scripts/train.py --algo='fe' --env='ImpossiblyGood-Construction-v0' --render --eval-argmax --eval-episodes=100 --procs=1 --eval-frequency=5000 --seed=4
