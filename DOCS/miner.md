# Miner Documentation

Complete guide for miners participating in the CancerAI subnet (netuid 76).

## Overview

The miner script enables you to:

- **Evaluate models locally** - Test your model against competition datasets
- **Self-check submitted models** - Verify your model using validator code
- **Upload to HuggingFace** - Publish your model and code
- **Submit to validators** - Register your model on-chain for evaluation

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Registration](#registration)
4. [Usage](#usage)
   - [Local Evaluation](#local-evaluation)
   - [Self-Check](#self-check)
   - [Upload to HuggingFace](#upload-to-huggingface)
   - [Submit to Validators](#submit-to-validators)
5. [Notes](#notes)

## Prerequisites

Before you begin, ensure you have:

- **Python 3.12** or higher
- **8GB RAM** minimum (for local model evaluation)
- **Virtual Environment** (recommended)
- **Bittensor Wallet** with registered hotkey on netuid 76
- **HuggingFace Account** (for model uploads)

## Installation

1. **Clone repository**

    ```bash
    git clone git@github.com:safe-scan-ai/cancer-ai.git
    cd cancer-ai
    ```

1. **Create a Virtual Environment**

    ```bash
    virtualenv venv --python=3.12
    source venv/bin/activate
    ```

1. **Install Required Python Packages**

    Install any required Python packages listed in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## Registration

If you haven't registered on the CancerAI subnet yet, follow these steps:

### 1. Create Wallet

```bash
# Create coldkey
btcli wallet new_coldkey --wallet.name miner

# Create hotkey
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

### 2. Register on Subnet

```bash
btcli subnet register \
    --netuid 76 \
    --subtensor.network finney \
    --wallet.name miner \
    --wallet.hotkey default
```

### 3. Verify Registration

```bash
btcli wallet overview --wallet.name miner
```

> **📚 Additional Resources**: Detailed registration instructions available at [Bittensor Docs](https://docs.learnbittensor.org/miners)

## Usage

### Setup Environment

Before running any commands:

```bash
# Navigate to project directory
cd cancer-ai-subnet

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:./"
```

---

### Local Evaluation

Test your ONNX model locally against competition datasets before submitting.

**What it does:**
- Downloads the latest competition dataset
- Runs your model through the evaluation pipeline
- Calculates all metrics including efficiency score
- Displays results in JSON format

**Command:**

```bash
python neurons/miner.py \
    --action evaluate \
    --competition_id <COMPETITION_ID> \
    --model_path <PATH_TO_MODEL.onnx>
```

**Arguments:**

| Argument | Description | Required | Default |
|----------|-------------|----------|----------|
| `--action` | Action to perform | Yes | - |
| `--model_path` | Path to ONNX model file | Yes | - |
| `--competition_id` | Competition identifier | Yes | - |
| `--clean-after-run` | Delete dataset after evaluation | No | false |
| `--dataset_dir` | Directory for datasets | No | `./datasets` |
| `--datasets_config_hf_repo_id` | HF repo for dataset config | No | `safescanai/competition-configuration` |

**Example:**

```bash
python neurons/miner.py \
    --action evaluate \
    --competition_id tricorder-3 \
    --model_path ./models/my_model.onnx \
    --clean-after-run
```

---

### Self-Check

Verify your submitted model by downloading it from the blockchain and evaluating it using the exact same code validators use.

**What it does:**
1. Retrieves your model metadata from the blockchain
2. Downloads your model from HuggingFace
3. Downloads the latest competition dataset
4. Evaluates using validator evaluation code
5. Displays comprehensive results:
   - All accuracy metrics (precision, recall, F1, etc.)
   - Efficiency score based on model size
   - Final competition score
   - Risk category breakdown (for Tricorder competitions)

**Command:**

```bash
python neurons/miner.py \
    --action self-check \
    --hotkey <YOUR_HOTKEY_SS58_ADDRESS> \
    --competition_id <COMPETITION_ID> \
    --subtensor.network finney
```

**Arguments:**

| Argument | Description | Required | Default |
|----------|-------------|----------|----------|
| `--action` | Must be `self-check` | Yes | - |
| `--hotkey` | Your miner's hotkey SS58 address | Yes | - |
| `--competition_id` | Competition identifier | Yes | - |
| `--subtensor.network` | Network (finney/test) | Yes | - |
| `--clean-after-run` | Delete dataset after evaluation | No | false |
| `--models.model_dir` | Directory for downloaded models | No | `/tmp/models` |
| `--models.dataset_dir` | Directory for datasets | No | `/tmp/datasets-extracted` |

**Example:**

```bash
python neurons/miner.py \
    --action self-check \
    --hotkey <YOUR_HOTKEY_SS58_ADDRESS> \
    --competition_id tricorder-3 \
    --subtensor.network finney
```

> **💡 Tip**: Use self-check before validators evaluate to catch issues early!

---


### Upload to HuggingFace

Compress and upload your model and code to HuggingFace.

**Requirements:**
- Repository must be type `model`
- Repository must be **public** (validators cannot access private repos)
- You need a HuggingFace API token with write permissions

**Command:**

```bash
python neurons/miner.py \
    --action upload \
    --competition_id <COMPETITION_ID> \
    --model_path <PATH_TO_MODEL.onnx> \
    --code_directory <CODE_DIRECTORY> \
    --hf_model_name <MODEL_FILENAME.onnx> \
    --hf_repo_id <USERNAME/REPO_NAME> \
    --hf_token <YOUR_HF_TOKEN>
```

**Arguments:**

| Argument | Description | Required |
|----------|-------------|----------|
| `--action` | Must be `upload` | Yes |
| `--competition_id` | Competition identifier | Yes |
| `--model_path` | Local path to ONNX model | Yes |
| `--code_directory` | Directory containing your code (exclude datasets) | Yes |
| `--hf_model_name` | Filename for model in HF repo | Yes |
| `--hf_repo_id` | HuggingFace repo (username/repo) | Yes |
| `--hf_token` | HuggingFace API token | Yes |

**Example:**

```bash
python neurons/miner.py \
    --action upload \
    --competition_id tricorder-3 \
    --model_path ./models/my_model.onnx \
    --code_directory ./my_code \
    --hf_model_name my_model.onnx \
    --hf_repo_id myusername/cancer-ai-model \
    --hf_token hf_xxxxxxxxxxxxx
```

---

### Submit to Validators

Register your model on-chain so validators can evaluate it.

**Requirements:**
- Model must already be uploaded to HuggingFace
- Repository must be **public**
- Code and model filenames must share the same base name

**Command:**

```bash
python neurons/miner.py \
    --action submit \
    --competition_id <COMPETITION_ID> \
    --hf_code_filename <CODE.zip> \
    --hf_model_name <MODEL.onnx> \
    --hf_repo_id <USERNAME/REPO_NAME> \
    --wallet.name <WALLET_NAME> \
    --wallet.hotkey <HOTKEY_NAME> \
    --netuid 76 \
    --subtensor.network finney
```

**Arguments:**

| Argument | Description | Required | Default |
|----------|-------------|----------|----------|
| `--action` | Must be `submit` | Yes | - |
| `--competition_id` | Competition identifier | Yes | - |
| `--hf_code_filename` | Zipped code filename in HF repo | Yes | - |
| `--hf_model_name` | Model filename in HF repo | Yes | - |
| `--hf_repo_id` | HuggingFace repo (username/repo) | Yes | - |
| `--wallet.name` | Coldkey name | Yes | - |
| `--wallet.hotkey` | Hotkey name | Yes | - |
| `--netuid` | Subnet ID | No | 76 |
| `--subtensor.network` | Network (finney/test) | Yes | - |

**Example:**

```bash
python neurons/miner.py \
    --action submit \
    --competition_id tricorder-3 \
    --hf_code_filename my_model.zip \
    --hf_model_name my_model.onnx \
    --hf_repo_id myusername/cancer-ai-model \
    --wallet.name miner \
    --wallet.hotkey default \
    --netuid 76 \
    --subtensor.network finney
```

> **⚠️ Important**: Code and model filenames must have matching base names (e.g., `model.onnx` and `model.zip`)

#### Post-Submission: Document Your Extrinsic

To prevent model copying, you **must** create an `extrinsic_record.json` file in your HuggingFace repo:

**1. Create the file:**

```json
{
    "hotkey": "<YOUR_HOTKEY_SS58_ADDRESS>",
    "extrinsic": "<EXTRINSIC_ID>"
}
```

**2. Find your extrinsic ID:**

1. Go to [Taostats Accounts](https://taostats.io/accounts)
2. Search for your hotkey SS58 address
3. Navigate to account dashboard → **Extrinsics**
4. Find the `Commitments.set_commitment` extrinsic
5. Copy the **Extrinsic ID**

**3. Upload to HuggingFace:**

Add `extrinsic_record.json` to the root of your model repository.

---

## Notes

### Model Requirements
- Models must be in **ONNX format**
- Model size affects efficiency score:
  - ≤50MB: Full efficiency score (1.0)
  - 50-150MB: Linear interpolation
  - ≥150MB: Zero efficiency score (0.0)

### Network Configuration
- **Mainnet**: `--subtensor.network finney --netuid 76`
- **Testnet**: `--subtensor.network test --netuid <testnet_id>`

### Resource Requirements
- Minimum 8GB RAM for local evaluation
- Sufficient disk space for datasets (~1-5GB per competition)

### Troubleshooting
- Ensure `PYTHONPATH` is set before running commands
- Verify virtual environment is activated
- Check that HuggingFace repositories are public
- Confirm wallet is registered on the subnet

### Getting Help
- [CancerAI Documentation](https://github.com/safe-scan-ai/cancer-ai/tree/main/DOCS)
- [Bittensor Discord](https://discord.gg/bittensor)
- [Competition Details](./COMPETITIONS.md)
