"""
Axon functionality for Cancer-AI validators.
Handles incoming requests from peer validators for result sharing.
"""

import bittensor as bt
from .protocol import CompetitionResultsSynapse


class ValidatorAxon:
    """Handles axon setup and message serving for validators."""
    
    def __init__(self, wallet, config, subtensor, metagraph=None, base_axon=None):
        self.wallet = wallet
        self.config = config
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.axon = base_axon  # Use the base validator's axon instead of creating new one
        # Storage for evaluation results that will be returned when requested
        self.stored_results = {}  # Format: {competition_id: {results: dict, ...}}
    
    def store_results(self, competition_id: str, results: dict, dataset_hash: str, model_count: int):
        """Store evaluation results for later sharing with peers."""
        import time
        self.stored_results[competition_id] = {
            'results': results,
            'dataset_hash': dataset_hash,
            'model_count': model_count,
            'timestamp': time.time()
        }
        bt.logging.info(f"Stored results for competition {competition_id}: {len(results)} miners")
    
    def handle_results_request(self, synapse: CompetitionResultsSynapse) -> CompetitionResultsSynapse:
        """
        Handle incoming results requests from other validators.
        Returns our evaluation results if available.
        """
        bt.logging.info(f"=== AXON: Received request from validator {synapse.validator_uid} ===")
        bt.logging.info(f"Competition: {synapse.competition_id}, Cycle: {synapse.cycle_id}")
        
        # Get our UID
        my_uid = -1
        if self.metagraph is not None:
            try:
                my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            except ValueError:
                bt.logging.error("Could not find our UID in metagraph")
                synapse.status = "failed"
                return synapse
        
        competition_id = synapse.competition_id
        
        # Check if we have results for this competition
        if competition_id in self.stored_results:
            stored = self.stored_results[competition_id]
            synapse.validator_uid = my_uid
            synapse.evaluation_results = stored['results']
            synapse.dataset_hash = stored['dataset_hash']
            synapse.model_count = stored['model_count']
            synapse.timestamp = stored['timestamp']
            synapse.status = "complete"
            bt.logging.info(f"Returning {len(synapse.evaluation_results)} results")
        else:
            synapse.validator_uid = my_uid
            synapse.status = "not_ready"
            synapse.evaluation_results = {}
            bt.logging.warning(f"No results for competition {competition_id}")
        
        return synapse
    
    def serve_axon(self):
        """Attach handler to existing axon (base validator's axon)."""
        bt.logging.info("=== Attaching P2P Handler to Validator Axon ===")
        
        if not self.axon:
            bt.logging.error("No axon available! Base validator axon must be created first.")
            raise RuntimeError("Base validator axon not available. Ensure base validator creates axon before ValidatorAxon initialization.")
        
        try:
            # Attach the results handler to the existing axon (already served by base validator)
            self.axon.attach(forward_fn=self.handle_results_request)
            bt.logging.info("CompetitionResultsSynapse handler attached to existing axon")
            
        except Exception as e:
            bt.logging.error(f"Failed to attach handler to axon: {e}")
            raise
    
    def start_axon_server(self):
        """Start the axon server to listen for incoming requests."""
        if self.axon:
            self.axon.start()
            bt.logging.info("Axon server started successfully - now listening for requests")
        else:
            bt.logging.error("Cannot start axon server - axon not initialized")