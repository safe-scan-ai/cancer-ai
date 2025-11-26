# Local Validator

The local validator allows you to:

- Run validation without registering a wallet on the subnet
- Skip wandb logging for local testing
- Use real subtensor to fetch models from metagraph
- Download datasets from HuggingFace
- Test validation logic locally

## Pre-requisites

- Dependencies as per [DOCS/validator.md](validator.md)
- HuggingFace token configured (optional but recommended)
- Create default wallet and hotkey using `btcli`


### Basic Command

```bash
source venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:./"

python neurons/validator.py \
  --wallet.name default \
  --wallet.hotkey default \
  --netuid 76 \
  --logging.debug \
  --wandb.off \
  --ignore_registered
```

### What Works

- ✅ Real subtensor connection to Finney network
- ✅ Fetch models from metagraph
- ✅ Download datasets from HuggingFace
- ✅ Run model evaluation
- ✅ Save results to CSV files
- ✅ No wandb logging
- ✅ No wallet registration required

### Limitations

- ❌ Cannot set weights (not registered)
- ❌ Cannot publish results (wandb disabled)
- ⚠️ Uses mock UID (-1) internally