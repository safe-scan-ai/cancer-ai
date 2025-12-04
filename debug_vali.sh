# mv ~/.bittensor/miners/default/default/netuid76/validator/state.json ~/.bittensor/miners/default/default/netuid76/validator/state_old.json

LOG_LOCATION="logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$LOG_LOCATION"

export PYTHONPATH="${PYTHONPATH}:./" && python neurons/validator.py --wallet.name default --wallet.hotkey omar__ --netuid=76 --logging.debug --wandb.off --ignore_registered --wandb.local_save 2>&1 | tee "$LOG_LOCATION/validator-debug.log"