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
    
    def __init__(self, dendrite, metagraph, wallet, config):
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.wallet = wallet
        self.config = config
    
    async def test_connection(self) -> bool:
        """
        Simple startup test - sends hello to discovered peer validators.
        Returns True if at least one peer responds.
        """
        bt.logging.info("🔗 P2P Test: Sending hello to peer validators...")
        
        try:
            my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            bt.logging.error("P2P Test: Could not find our UID")
            return False
        
        # Get peer validators (dynamic discovery)
        peer_uids = self.get_validator_uids()
        
        if not peer_uids:
            bt.logging.warning("P2P Test: No peer validators found")
            return False
        
        bt.logging.info(f"P2P Test: Found {len(peer_uids)} peers: {peer_uids}")
        
        # Send simple hello synapse
        synapse = CompetitionResultsSynapse(
            validator_uid=my_uid,
            competition_id="hello",
            cycle_id="startup-test",
            timestamp=time.time(),
            evaluation_results={},
            status="request"
        )
        
        axons = [self.metagraph.axons[uid] for uid in peer_uids]
        
        try:
            responses = await self.dendrite(axons, synapse, deserialize=True, timeout=10.0)
            
            success_count = 0
            for i, response in enumerate(responses):
                uid = peer_uids[i]
                if response is not None:
                    bt.logging.info(f"P2P Test: ✅ UID {uid} responded (status: {response.status})")
                    success_count += 1
                else:
                    bt.logging.warning(f"P2P Test: ❌ UID {uid} no response")
            
            bt.logging.info(f"P2P Test: {success_count}/{len(peer_uids)} peers reachable")
            return success_count > 0
            
        except Exception as e:
            bt.logging.error(f"P2P Test failed: {e}")
            return False
    
    def get_validator_uids(self) -> List[int]:
        """
        Get UIDs of other validators on the network.
        Uses dynamic discovery from metagraph - finds validators with active axons.
        """
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
                    bt.logging.info(f"Using custom validator UIDs: {custom_uids}")
                    return [uid for uid in custom_uids if uid != my_uid]
            except ValueError:
                bt.logging.warning(f"Invalid test_validator_uids format: {test_uids_str}")
        
        # Dynamic discovery - find validators with active axons
        validator_uids = []
        for uid in range(len(self.metagraph.axons)):
            if uid == my_uid:
                continue
            
            # Check if this UID has validator permit
            if uid < len(self.metagraph.validator_permit):
                if self.metagraph.validator_permit[uid]:
                    axon = self.metagraph.axons[uid]
                    # Check if axon is reachable (has valid IP/port)
                    if axon.ip and axon.port and axon.ip not in ['0.0.0.0', '']:
                        validator_uids.append(uid)
        
        return validator_uids
    
    async def collect_results(
        self,
        competition_id: str,
        cycle_id: str,
        timeout: float = 30.0
    ) -> Dict[int, dict]:
        """
        Collect evaluation results from peer validators.
        
        Args:
            competition_id: ID of the competition
            cycle_id: Unique cycle identifier
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary mapping validator UID to their results
        """
        bt.logging.info("=== P2P: Collecting results from peer validators ===")
        
        # Get my UID
        try:
            my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            bt.logging.error("Could not find our UID")
            return {}
        
        # Get available validator UIDs (auto-discovery)
        validator_uids = self.get_validator_uids()
        
        if not validator_uids:
            bt.logging.warning("P2P: No peer validators available")
            return {}
        
        bt.logging.info(f"P2P: Querying {len(validator_uids)} validators: {validator_uids}")
        
        # Create request synapse
        synapse = CompetitionResultsSynapse(
            validator_uid=my_uid,
            competition_id=competition_id,
            cycle_id=cycle_id,
            timestamp=time.time(),
            evaluation_results={},
            status="request"
        )
        
        axons = [self.metagraph.axons[uid] for uid in validator_uids]
        responses = await self.dendrite(
            axons,
            synapse,
            deserialize=True,
            timeout=timeout
        )
        
        # Process responses
        collected_results = {}
        for i, response in enumerate(responses):
            uid = validator_uids[i]
            
            if response is not None and response.status == "complete":
                collected_results[uid] = {
                    'results': response.evaluation_results,
                    'dataset_hash': response.dataset_hash,
                    'model_count': response.model_count,
                    'timestamp': response.timestamp
                }
                bt.logging.info(f"✅ UID {uid}: {len(response.evaluation_results)} results")
            elif response is not None and response.status == "not_ready":
                bt.logging.info(f"⏳ UID {uid}: not ready")
            else:
                bt.logging.warning(f"❌ UID {uid}: no response")
        
        bt.logging.info(f"P2P: Collected from {len(collected_results)}/{len(validator_uids)} validators")
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