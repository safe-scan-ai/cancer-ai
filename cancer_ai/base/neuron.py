# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import copy
import sys
import random
import time
import sys

import bittensor as bt

from abc import ABC, abstractmethod

# Sync calls set weights and also resyncs the metagraph.
from ..utils.config import check_config, add_args, path_config
from ..utils.misc import ttl_get_block
from .. import __spec_version__ as spec_version
from ..mock import MockSubtensor, MockMetagraph


class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    neuron_type: str = "BaseNeuron"

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return path_config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = spec_version

    @property
    def block(self):
        return ttl_get_block(self)

    def __init__(self, config=None):
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)

        # Set up logging with the provided configuration.
        bt.logging.set_config(config=self.config.logging)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.
        if self.config.mock:
            self.wallet = bt.MockWallet(config=self.config)
            self.subtensor = MockSubtensor(self.config.netuid, wallet=self.wallet)
            self.metagraph = MockMetagraph(self.config.netuid, subtensor=self.subtensor)
        else:
            self.wallet = bt.wallet(config=self.config)
            self.subtensor = bt.subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)

        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Check if the miner is registered on the Bittensor network before proceeding further.
        self.check_registered()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
        )
        self.step = 0

        self._last_updated_block = self.metagraph.last_update[self.uid]

    @abstractmethod
    def run(self): ...

    def sync(self, retries=5, delay=10, force_sync=False):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        attempt = 0
        while attempt < retries:
            try: 
                # Ensure miner or validator hotkey is still registered on the network.
                self.check_registered()
                if self.config.filesystem_evaluation:
                        break
                if self.should_sync_metagraph() or force_sync:
                    bt.logging.info("Resyncing metagraph in progress.")
                    self.resync_metagraph(force_sync=True)
                    self.save_state()

                if self.should_set_weights():
                    self.set_weights()
                    self._last_updated_block = self.block
                    self.save_state()

                break

            except BrokenPipeError as e:
                attempt += 1
                bt.logging.error(f"BrokenPipeError: {e}. Retrying...")
                time.sleep(delay)

            except Exception as e:
                attempt += 1
                bt.logging.error(f"Unexpected error occurred: {e}. Retrying...")
                time.sleep(delay)

        if attempt == retries:
            bt.logging.error(
                "Failed to sync metagraph after %d retries; exiting.",
                retries,
            )
            sys.exit(1)

    def check_registered(self):
        retries = 3
        while retries > 0:
            try:
                if not hasattr(self, "is_registered"):
                    self.is_registered = self.subtensor.is_hotkey_registered(
                        netuid=self.config.netuid,
                        hotkey_ss58=self.wallet.hotkey.ss58_address,
                    )
                    if not self.is_registered:
                        bt.logging.error(
                            f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                            f" Please register the hotkey using `btcli subnets register` before trying again"
                        )
                        sys.exit()

                return self.is_registered

            except Exception as e:
                bt.logging.error(f"Error checking validator's hotkey registration: {e}")
                retries -= 1
                if retries == 0:
                    sys.exit()
                else:
                    bt.logging.info(f"Retrying... {retries} retries left.")

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """

        elapsed = self.block - self._last_updated_block

        # Only set weights if epoch has passed
        return elapsed > self.config.neuron.epoch_length
    
    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        elapsed = self.block - self._last_updated_block

        # Only set weights if epoch has passed and this isn't a MinerNeuron.
        return elapsed > self.config.neuron.epoch_length and self.neuron_type != "MinerNeuron"
