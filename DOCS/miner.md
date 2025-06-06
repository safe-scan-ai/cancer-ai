# Miner Script Documentation

This documentation provides an overview of the miner script, its functionality, requirements, and usage instructions.

## Overview

The miner script is designed to manage models, evaluate them locally, and upload them to HuggingFace, as well as submit models to validators within a specified network.

Key features of the script include:

- **Local Model Evaluation**: Allows you to evaluate models against a dataset locally.
- **HuggingFace Upload**: Compresses and uploads models and code to HuggingFace.
- **Model Submission to Validators**: Saves model information in the metagraph, enabling validators to test the models.

## Prerequisites

- **Python 3.12**: The script is written in Python and requires Python 3.12 to run.
- **Virtual Environment**: It's recommended to run the script within a virtual environment to manage dependencies.
- **8GB RAM**: minimum required operating memory for testing (evaluate) machine learning model locally

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

## Registering miner on the subnet

If you haven't yet created a miner wallet and registered on our subnet here is the set of commands to run:

Create a miner coldkey:

```
btcli wallet new_coldkey --wallet.name miner
```

Create a hotkey for the miner:
```
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

Register miner on the CancerAI subnet:
```
btcli subnet recycle_register --netuid <Cancer AI subnet id> --subtensor.network finney --wallet.name miner --wallet.hotkey default
```

Check that your key was registered:
```
btcli wallet overview --wallet.name miner 
```

## Usage

### Prerequisites

Before running the script, ensure the following:

- You are in the base directory of the project.
- Your virtual environment is activated.
- Run the following command to set the `PYTHONPATH`:

```
export PYTHONPATH="${PYTHONPATH}:./"
```

### Evaluate Model Locally

This mode performs the following tasks:

- Downloads the dataset.
- Loads your model.
- Prepares data for execution.
- Logs evaluation results.

To evaluate a model locally, use the following command:

```
python neurons/miner.py --action evaluate --competition_id <COMPETITION ID> --model_path <NAME OF FILE WITH EXTENSION>
```

Command line argument explanation

- `--action` - action to perform, choices are "upload", "evaluate", "submit"
- `--model_path` - local path of ONNX model
- `--competition_id` - ID of competition. List of current competitions are in [competition_config.json](config/competition_config.json)
- `--clean-after-run` - it will delete dataset after evaluating the model
- `--model_dir` - path for storing models (default: "./models")
- `--dataset_dir` - path for storing datasets (default: "./datasets")
- `--datasets_config_hf_repo_id` - hugging face repository ID for datasets configuration - ex. "safescanai/competition-configuration-testnet" in case of testnet

### Upload to HuggingFace

This mode compresses the code provided by `--code-path` and uploads the model and code to HuggingFace.
Repository ID should be a repository type "model".

The repository needs to be public for validator to pick it up.

To upload to HuggingFace, use the following command:

```bash
python neurons/miner.py \
    --action upload \
    --competition_id <COMPETITION ID> \
    --model_path <NAME OF FILE WITH EXTENSION> \
    --code_directory <CODE DIRECTORY WITHOUT DATASETS> \
    --hf_model_name <MODEL NAME WITH EXTENSION> \
    --hf_repo_id <HF REPO ID> \
    --hf_token <HF API TOKEN>
```

Command line argument explanation

- `--code_directory` - local directory of code
- `--hf_repo_id` - hugging face repository ID - ex. "username/repo"
- `--hf_token` - hugging face authentication token
- `--hf_model_name` - name of file to store in hugging face repository

### Submit Model to Validators

This mode saves model information in the metagraph, allowing validators to retrieve information about your model for testing.

The repository you are submitting needs to be public for validator to pick it up.

To submit a model to validators, use the following command:

```bash
python neurons/miner.py \
    --action submit \
    --competition_id melanoma-1\
    --hf_code_filename skin_melanoma_small.zip\
    --hf_model_name best_model.onnx \
    --hf_repo_id safescanai/test_dataset \
    --wallet.name miner2 \
    --wallet.hotkey default \
    --netuid 163 \
    --subtensor.network test
```

Command line argument explanation

- `--hf_code_filename` - name of file in hugging face repository containing zipped code
- `--hf_model_name` - name of file in hugging face repository containing model
- `--wallet.name` - name of wallet coldkey used for authentication with Bittensor network
- `--wallet.hotkey` - name of wallet hotkey used for authentication with Bittensor network
- `--netuid` - subnet number
- `--subtensor.network` - Bittensor network to connect to - <test|finney>

## Notes

- **Environment**: The script uses the environment from which it is executed, so ensure all necessary environment variables and dependencies are correctly configured.
- **Model Evaluation**: The `evaluate` action downloads necessary datasets and runs the model locally; ensure that your local environment has sufficient resources.
