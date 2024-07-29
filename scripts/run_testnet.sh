python neurons/validator.py --netuid 163 --subtensor.network test --wallet.name miner --wallet.hotkey hot_validator --logging.debug


pm2 start neurons/miner.py --interpreter python3 --  --netuid 163 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug
pm2 start neurons/miner.py --interpreter python3 --  --netuid 163 --subtensor.network test --wallet.name miner --wallet.hotkey second_miner --logging.debug

pm2 start neurons/miner.py --interpreter python3 --name researcher --  --netuid 163 --subtensor.network test --wallet.name miner --wallet.hotkey second_miner --logging.debug --researcher

pm2 start neurons/miner.py --interpreter python3 --name miner3 --  --netuid 163 --subtensor.network test --wallet.name miner --wallet.hotkey third_miner --logging.debug 


pm2 start neurons/miner.py --interpreter python3 --name miner3 --  --netuid 163 --subtensor.network test --wallet.name miner --wallet.hotkey miner_testowy --logging.debug 

btcli subnet register --netuid 163 --wallet.name miner --wallet.hotkey second_miner --subtensor.network test
btcli subnet register --netuid 163 --wallet.name miner --wallet.hotkey third_miner --subtensor.network test
btcli subnet register --netuid 163 --wallet.name miner --wallet.hotkey miner_testowy --subtensor.network test


btcli wallet new_hotkey --wallet.name miner
