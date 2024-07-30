### How to run Validator

1. Contact us on [Safe Scan Discord, #validator-support](https://discord.gg/PZevpMJrfW) to request the Dataset API key and Stats API key which will enable you to pull resources for testing researchers and generation of synthetic queries.
2. Setup config parameters which can be found in ./neurons/cancer_ai/utils/config.py. You can either provide these parameters with the flags when running the validator.py or adjust the default values for the config parameters in the config.py file directly. 
The previous step api keys are the only ones required to set manually with flags or in config.py. The Validator will work with default settings for the rest of the parameters.
3. Run the validator.py with our subnet <subnet_id> flag. Here is an example on how to run it on our testnet:

```
pm2 start neurons/validator.py --interpreter python3 -- --netuid 163 --subtensor.network test --wallet.name <validator-wallet-name> --wallet.hotkey <validator-wallet-hotkey> --logging.debug
```