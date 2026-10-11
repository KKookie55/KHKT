#!/bin/bash
cd /home/z/my-project
for SEED in 777 2024 31337; do
    echo "==== Starting seed $SEED at $(date) ===="
    timeout 580 python3 -u scripts/train_v85.py --seeds $SEED --horizons 1 --splits all --models all 2>&1 | tail -10
    echo "==== Done seed $SEED at $(date) ===="
done
echo "==== ALL SEEDS DONE ===="
ls -la /home/z/my-project/download/real_data_v85/runs/ | wc -l
