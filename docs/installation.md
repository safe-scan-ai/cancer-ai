# Cancer AI  installation Guide 🚀

## Requirements

Before installing Bittensor, ensure you have the following requirements:

- Python 3.10 - 3.12 with `virtualenv`
- `pip` (Python package installer) 📦
- `git` 🛠️
- At least 2GB of RAM
- Prefered `Ubuntu 24.04 LTS` server, though it might work somewhere else

## Installation

1. Clone the repository into your local machine by running the following command:

```sh
git clone https://github.com/safe-scan-ai/cancer-ai.git
cd cancer-ai
```

2. Set up virtual environment and install dependencies

```
virtualenv venv --python=3.10
source venv/bin/activate
pip install -r requirements.txt
```

3. Install PM2 for process management

```bash
sudo npm install pm2 -g
pm2 --version # verify version
```

## Registration

Cancer AI is Subnet **163** on the testnet.

To register your wallets onto the subnet, you can run:

<strong>Mainnet</strong>
```bash
btcli subnet register --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey_name> --netuid XXXX
```

<strong>Testnet</strong>
```bash
btcli subnet register --wallet.name <your_wallet_name> --wallet.hotkey <your_hotkey_name> --netuid 163 --subtensor.network test
```


## Troubleshooting

If you encounter issues during installation,  reach out to the Safe Scan team on Discord for further assistance.

## Next Steps

- [Run a Validator](./validator.md)
- [Run a Miner](./miner.md)
- [Run a Researcher](./researcher.md)
