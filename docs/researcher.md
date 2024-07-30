### Researcher

Note that if you are planning to participate as the Researcher it is adviced to run the Researcher Miner in the immunity period (right after registration). The Researcher Miner is at the same time performing Regular Miner job, but as the reward system is based on the response pace it is possible that the general score of the Miner will drop if you are executing both Researcher and Miner tasks on the same machine.
Hence, if you are already running a Regular Miner successfully after the immunity period it is also adviced to adjust the miner.py to handle the Researcher tasks asynchronously and/or introducing proxy for handling the Researcher processing on another machine.

1. Setup config parameters which can be found in ./neurons/cancer_ai/utils/config.py. You can either provide these parameters with the flags when running the miner.py or adjust the default values for the config parameters in the config.py file directly. A miner can be both a Regular miner (offering computational power) and the Researcher. 
To run as the Researcher Miner make sure that the --researcher flag is enabled and that the --testing_session_id is set to valid uuid. As you are testing different models provide different testing_session_id for each model. Predictions are later on aggregated on [Statistics API](https://statistics.safe-scan.ai/) based on this testing_session_id.
v
2. Run the miner.py script with our subnet <subnet_id> flag. Here is an example on how to run it on our testnet:

```
pm2 start neurons/miner.py --interpreter python3 -- --netuid 163 --subtensor.network test --wallet.name <miner-wallet-name> --wallet.hotkey <miner-wallet-hotkey> --logging.debug --researcher --testing_session_id <valid-uuid>
```
3. When the testing is done the outcome can be found on [Statistics API](https://statistics.safe-scan.ai/). Your Researcher Miner will also receive a feedback with your models prediction, base models prediction and the actual answer for the task. You can find it under forward_get_feedback function in miner.py. If it appears that your Researcher models is better then our current model, reach us on Discord. We will then test your model outside of the subnet to confirm on its accuracy and hopefully introduce it as a new base model for the Subnet and the Skinscan App!