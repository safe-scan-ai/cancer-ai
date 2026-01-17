"""
P2P result collection for Cancer-AI validators.
Queries peer validators for their evaluation results.
Based on subnet-bizantyne-prototype P2P communication pattern.
"""

import time
import asyncio
import bittensor as bt
from typing import Dict, List, Optional
from cancer_ai.protocol import CompetitionResultsSynapse


class P2PCollector:
    """Collects evaluation results from peer validators."""
    
    # Hardcoded test validator UIDs
    TEST_VALIDATOR_UIDS = [83, 84]
    
    def __init__(self, dendrite, metagraph, wallet, config):
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.wallet = wallet
        self.config = config
    
    def get_validator_uids(self, use_test_uids: bool = False) -> List[int]:
        """
        Get UIDs of other validators on the network.
        
        Args:
            use_test_uids: If True, use hardcoded TEST_VALIDATOR_UIDS (like bizantyne)
        """
        my_uid = -1
        try:
            my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            bt.logging.error("Could not find our UID")
            return []
        
        # Check for custom test UIDs from config
        test_uids_str = getattr(self.config, 'test_validator_uids', '')
        if test_uids_str:
            try:
                custom_uids = [int(uid.strip()) for uid in test_uids_str.split(',') if uid.strip()]
                if custom_uids:
                    bt.logging.info(f"Using custom test validator UIDs: {custom_uids}")
                    return [uid for uid in custom_uids if uid != my_uid]
            except ValueError:
                bt.logging.warning(f"Invalid test_validator_uids format: {test_uids_str}")
        
        # Use hardcoded test UIDs 
        if use_test_uids or getattr(self.config, 'mock_p2p_test', False):
            test_uids = [uid for uid in self.TEST_VALIDATOR_UIDS if uid != my_uid]
            bt.logging.info(f"Using hardcoded test validator UIDs: {test_uids} (excluding self: {my_uid})")
            
            # Filter for validators that actually exist in metagraph and have valid axons
            available_uids = []
            for uid in test_uids:
                if uid >= len(self.metagraph.axons):
                    bt.logging.warning(f"  UID {uid}: Out of range (metagraph has {len(self.metagraph.axons)} axons)")
                    continue
                    
                axon = self.metagraph.axons[uid]
                bt.logging.info(f"  Checking UID {uid}: IP={axon.ip}, Port={axon.port}")
                
                if axon.ip is None or axon.port is None:
                    bt.logging.warning(f"    ❌ UID {uid} has no IP/port")
                    continue
                    
                if axon.ip in ['0.0.0.0', '']:
                    bt.logging.warning(f"    ❌ UID {uid} has invalid IP: {axon.ip}")
                    continue
                
                available_uids.append(uid)
                bt.logging.info(f"    ✅ UID {uid} is available!")
            
            return available_uids
        
        # Dynamic discovery (production mode)
        validator_uids = []
        for uid in range(len(self.metagraph.axons)):
            if uid == my_uid:
                continue
            
            # Check if this is a validator
            if uid < len(self.metagraph.validator_permit):
                if self.metagraph.validator_permit[uid]:
                    axon = self.metagraph.axons[uid]
                    # Check if axon is reachable
                    if axon.ip and axon.port and axon.ip not in ['0.0.0.0', '']:
                        validator_uids.append(uid)
        
        return validator_uids
    
    async def collect_results(
        self,
        competition_id: str,
        cycle_id: str,
        timeout: float = 30.0,
        use_test_uids: bool = False
    ) -> Dict[int, dict]:
        """
        Collect evaluation results from peer validators.
        Like bizantyne prototype's run_collection_phase().
        
        Args:
            competition_id: ID of the competition
            cycle_id: Unique cycle identifier
            timeout: Request timeout in seconds
            use_test_uids: Use hardcoded test UIDs
            
        Returns:
            Dictionary mapping validator UID to their results
        """
        print("=" * 60)
        print("=== PHASE 2: COLLECTING RESULTS FROM PEERS ===")
        print("=" * 60)
        bt.logging.info("=" * 60)
        bt.logging.info("=== PHASE 2: COLLECTING RESULTS FROM PEERS ===")
        bt.logging.info("=" * 60)
        
        # Get my UID
        try:
            my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            print("ERROR: Could not find our UID")
            bt.logging.error("Could not find our UID")
            return {}
        
        # Get available validator UIDs (use test mode if configured)
        validator_uids = self.get_validator_uids(use_test_uids=use_test_uids)
        
        if not validator_uids:
            bt.logging.warning("No peer validators available to query")
            return {}
        
        bt.logging.info(f"🔍 Querying {len(validator_uids)} validators: {validator_uids}")
        
        # Create request synapse
        synapse = CompetitionResultsSynapse(
            validator_uid=my_uid,
            competition_id=competition_id,
            cycle_id=cycle_id,
            timestamp=time.time(),
            evaluation_results={},
            status="request"
        )
        
        # Get axons
        axons = [self.metagraph.axons[uid] for uid in validator_uids]
        
        # Query validators
        print("📡 Sending results requests to peer validators...")
        bt.logging.info("📡 Sending results requests to peer validators...")
        responses = await self.dendrite(
            axons,
            synapse,
            deserialize=True,
            timeout=timeout
        )
        
        # Process responses (like bizantyne prototype)
        collected_results = {}
        for i, response in enumerate(responses):
            uid = validator_uids[i]
            
            # Debug: Log what we actually received
            print(f"Response from UID {uid}:")
            bt.logging.info(f"Response from UID {uid}:")
            
            if response is not None:
                print(f"   - Response type: {type(response)}")
                print(f"   - Status: {response.status}")
                print(f"   - Validator UID: {getattr(response, 'validator_uid', 'N/A')}")
                print(f"   - Competition ID: {getattr(response, 'competition_id', 'N/A')}")
                print(f"   - Has evaluation_results: {hasattr(response, 'evaluation_results')}")
                bt.logging.info(f"   - Status: {response.status}")
                bt.logging.info(f"   - Validator UID: {getattr(response, 'validator_uid', 'N/A')}")
                
                if hasattr(response, 'evaluation_results'):
                    results_count = len(response.evaluation_results) if response.evaluation_results else 0
                    print(f"   - Results count: {results_count}")
                    bt.logging.info(f"   - Results count: {results_count}")
                    if response.evaluation_results:
                        sample = list(response.evaluation_results.items())[:3]
                        print(f"   - Sample results: {sample}")
                
                # Check dendrite status
                if hasattr(response, 'dendrite') and response.dendrite:
                    print(f"   - Dendrite status_code: {response.dendrite.status_code}")
                    print(f"   - Dendrite status_message: {response.dendrite.status_message}")
                    bt.logging.info(f"   - Dendrite status_code: {response.dendrite.status_code}")
            else:
                print(f"   - Response is None!")
                bt.logging.warning(f"   - Response is None!")
            
            if response is not None and response.status == "complete":
                collected_results[uid] = {
                    'results': response.evaluation_results,
                    'dataset_hash': response.dataset_hash,
                    'model_count': response.model_count,
                    'timestamp': response.timestamp
                }
                bt.logging.info(f"✅ Collected results from UID {uid}: {len(response.evaluation_results)} miners")
            elif response is not None and response.status == "not_ready":
                bt.logging.warning(f"⚠️ UID {uid} not ready yet (evaluation still running)")
            else:
                bt.logging.warning(f"❌ Failed to get results from UID {uid}: status={response.status if response else 'None'}")
        
        bt.logging.info(f"Collected results from {len(collected_results)}/{len(validator_uids)} validators")
        
        # Log summary
        if collected_results:
            print("=== COLLECTED RESULTS SUMMARY ===")
            bt.logging.info("=== COLLECTED RESULTS SUMMARY ===")
            for uid, data in collected_results.items():
                bt.logging.info(f"  Validator {uid}: {data['model_count']} miners")
        
        bt.logging.info("=" * 60)
        return collected_results


class ResultAggregator:
    """Aggregates results from multiple validators."""
    
    @staticmethod
    def aggregate_scores(
        local_results: Dict[str, float],
        peer_results: Dict[int, dict],
        min_validators: int = 2
    ) -> Dict[str, float]:
        """
        Aggregate scores from local and peer validators.
        
        Uses weighted average or median depending on validator count.
        """
        if not peer_results:
            bt.logging.info("No peer results, using local results only")
            return local_results
        
        # Collect all scores per miner
        all_scores = {} 
        
        # Add local results
        for hotkey, score in local_results.items():
            if hotkey not in all_scores:
                all_scores[hotkey] = []
            all_scores[hotkey].append(score)
        
        # Add peer results
        for uid, data in peer_results.items():
            for hotkey, score in data['results'].items():
                if hotkey not in all_scores:
                    all_scores[hotkey] = []
                all_scores[hotkey].append(score)
        
        # Calculate aggregated scores (median)
        import numpy as np
        aggregated = {}
        for hotkey, scores in all_scores.items():
            if len(scores) >= min_validators:
                aggregated[hotkey] = float(np.median(scores))
            else:
                # Not enough validators, use mean
                aggregated[hotkey] = float(np.mean(scores))
        
        bt.logging.info(f"Aggregated scores for {len(aggregated)} miners from {len(peer_results) + 1} validators")
        
        return aggregated