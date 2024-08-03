### How to run Miner

1. Setup config parameters which can be found in ./neurons/cancer_ai/utils/config.py. You can either provide these parameters with the flags when running the miner.py or adjust the default values for the config parameters in the config.py file directly. A miner can be both a Regular miner (offering computational power) and the Researcher. To proceed with just Regular miner (without announcing to the subnet participation as a Researcher) make sure that you run the miner without --researcher flag.
The Regular Miner should be able to run right away with default config parameters.

2. Run the miner.py script with our subnet <subnet_id> flag.  Run the validator.py with our subnet <subnet_id> flag. Here is an example on how to run it on our testnet:

```
pm2 start neurons/miner.py --interpreter python3 -- --netuid 163 --subtensor.network test --wallet.name <miner-wallet-name> --wallet.hotkey <miner-wallet-hotkey> --logging.debug
```