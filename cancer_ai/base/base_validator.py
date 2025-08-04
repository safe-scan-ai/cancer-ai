# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

from abc import abstractmethod

import sys
import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt

from typing import Union
from traceback import print_exception

from .neuron import BaseNeuron
from .utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)
from ..mock import MockDendrite
from ..utils.config import add_validator_args

from cancer_ai.validator.rewarder import CompetitionResultsStore
from cancer_ai.validator.models import OrganizationDataReferenceFactory
from .. import __spec_version__ as spec_version


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None, exit_event: threading.Event = None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.organizations_data_references = OrganizationDataReferenceFactory.get_instance()
        self.competition_results_store = CompetitionResultsStore()
        self.org_latest_updates = {}
        # add log with file path for loading state 
        state_file_path = self.config.neuron.full_path + "/state.json"
        bt.logging.info(f"Loading state from {state_file_path}")
        self.load_state()
        # Init sync with the network. Updates the metagraph.
        self.sync(force_sync=True)

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()
        self.exit_event = exit_event

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    @abstractmethod
    def concurrent_forward(self):
        pass

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1
        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"VALIDATOR FAILURE: Error during validation: {str(err)}")
            bt.logging.error(f"Error type: {type(err).__name__}")
            bt.logging.error(f"Error occurred in method: {self.concurrent_forward.__name__}")
            bt.logging.error(f"Current step: {self.step}")
            
            # Log the full stack trace
            import traceback
            stack_trace = traceback.format_exc()
            bt.logging.error(f"Full stack trace:\n{stack_trace}")
            bt.logging.error(str(print_exception(type(err), err, err.__traceback__)))
            
            # Log additional context information
            bt.logging.error(f"Validator state: running={self.is_running}, should_exit={self.should_exit}")
            
            if self.exit_event:
                bt.logging.error("Setting exit event and terminating validator", exc_info=True)
                self.exit_event.set()
            sys.exit(1)

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        bt.logging.info(f"run_in_background_thread called with is_running={self.is_running}")
        
        # Get the current call stack to see what's calling run_in_background_thread
        import traceback
        stack_trace = traceback.format_stack()
        bt.logging.info(f"Call stack for run_in_background_thread:\n{''.join(stack_trace)}")
        
        if not self.is_running:
            bt.logging.info("Starting validator in background thread.")
            self.should_exit = False
            bt.logging.info(f"Set should_exit to {self.should_exit}, creating thread")
            self.thread = threading.Thread(target=self.run, daemon=True)
            bt.logging.info(f"Starting thread with daemon={self.thread.daemon}")
            self.thread.start()
            self.is_running = True
            bt.logging.info(f"Thread started, set is_running to {self.is_running}")
            bt.logging.info("Validator started successfully in background thread")
        else:
            bt.logging.warning("Attempted to start validator that is already running")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        bt.logging.info(f"stop_run_thread called with is_running={self.is_running}")
        import traceback
        stack_trace = traceback.format_stack()
        bt.logging.info(f"Call stack for stop_run_thread:\n{''.join(stack_trace)}")
        
        if self.is_running:
            bt.logging.info("Stopping validator in background thread.")
            self.should_exit = True
            bt.logging.info(f"Set should_exit to {self.should_exit}, joining thread")
            self.thread.join(5)
            self.is_running = False
            bt.logging.info(f"Thread joined, set is_running to {self.is_running}")
            bt.logging.info("Validator stopped successfully")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback_obj: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        bt.logging.info(f"__exit__ called with exc_type={exc_type}, exc_value={exc_value}")
        
        # Get the current call stack to see what's calling __exit__
        import traceback
        stack_trace = traceback.format_stack()
        bt.logging.info(f"Call stack for __exit__:\n{''.join(stack_trace)}")
        
        # If there's an exception, log it
        if exc_type is not None:
            bt.logging.error(f"Exception in context: {exc_type.__name__}: {exc_value}")
            if traceback_obj:
                bt.logging.error(f"Exception traceback: {''.join(traceback.format_tb(traceback_obj))}")
        
        if self.is_running:
            bt.logging.info("Stopping validator in background thread from __exit__ method.")
            self.should_exit = True
            bt.logging.info(f"Set should_exit to {self.should_exit}, joining thread")
            self.thread.join(5)
            self.is_running = False
            bt.logging.info(f"Thread joined, set is_running to {self.is_running}")
            bt.logging.info("Validator stopped successfully from __exit__ method")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        # test mode, don't commit weights
        if self.config.filesystem_evaluation:
            bt.logging.debug("Skipping settings weights in filesystem evaluation mode")
            return

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = self.scores / norm

        # Ensure UID 0 gets 100% of weights for burning
        # Store the original weight for UID 0 for logging
        original_uid0_weight = raw_weights[0]
        
        # Set UID 0 to 100%
        raw_weights[0] = 1.0
        
        # Set all other UIDs to 0
        raw_weights[1:] = 0.0
        
        bt.logging.info(f"Set UID 0 weight from {original_uid0_weight:.4f} to {raw_weights[0]:.4f} (100%)")

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Verify UID 0 weight after processing
        if 0 in processed_weight_uids:
            uid0_index = np.where(processed_weight_uids == 0)[0][0]
            uid0_processed_weight = processed_weights[uid0_index]
            total_processed_weight = np.sum(processed_weights)
            uid0_percentage = (uid0_processed_weight / total_processed_weight) * 100 if total_processed_weight > 0 else 0
            bt.logging.info(f"UID 0 weight after processing: {uid0_processed_weight:.4f} ({uid0_percentage:.1f}% of total)")

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=spec_version
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self, force_sync=False):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph() validator")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons and not force_sync:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    @abstractmethod
    def save_state(self):
        """Saves the state of the validator to a file."""
       
    @abstractmethod
    def load_state(self):
        """Loads the state of the validator from a file."""
