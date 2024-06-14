module.exports = {
    apps : [{
      name: "validator",
      script: "neurons/validator.py",
      args: "--netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9944 --wallet.name validator --wallet.hotkey default --logging.debug --neuron.epoch_length 5",
      cwd: "home/tensor/cancer-ai-clone",
      interpreter: "/.venv/bin/python",
      watch: false,
      env: {
        PYTHONPATH: process.env.PYTHONPATH + ":/home/tensor/cancer-ai-clone/template"
      }
    }]
  };