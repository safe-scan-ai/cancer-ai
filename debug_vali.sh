mv \
  /Users/wojtasy/.bittensor/miners/2025-miner/2025-miner-hotkey/netuid76/validator/state.json \
  /Users/wojtasy/.bittensor/miners/2025-miner/2025-miner-hotkey/netuid76/validator/state_a.json

LOG_LOCATION="logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$LOG_LOCATION"

export PYTHONPATH="${PYTHONPATH}:./" && \
  ./venv/bin/python neurons/validator.py \
    --wallet.name 2025-miner \
    --wallet.hotkey 2025-miner-hotkey \
    --netuid=76 \
    --logging.trace \
    --wandb.off \
    --ignore_registered \
    --wandb.local_save \
    2>&1 | tee "$LOG_LOCATION/validator-debug.log"