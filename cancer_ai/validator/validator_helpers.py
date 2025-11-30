"""Helper functions for validator to reduce main file size."""

import json
import time
import bittensor as bt

from cancer_ai.utils.config import BLACKLIST_FILE_PATH, BLACKLIST_FILE_PATH_TESTNET


def should_refresh_miners(last_miners_refresh, miners_refresh_interval) -> bool:
    """Check if miners should be refreshed based on time interval."""
    if last_miners_refresh is None:
        return True
    time_since_refresh = time.time() - last_miners_refresh
    return time_since_refresh >= miners_refresh_interval * 60


def load_blacklisted_hotkeys(test_mode: bool) -> set:
    """Load blacklisted hotkeys from file."""
    blacklist_file = (
        BLACKLIST_FILE_PATH_TESTNET if test_mode else BLACKLIST_FILE_PATH
    )
    with open(blacklist_file, "r", encoding="utf-8") as f:
        return set(json.load(f))


async def retrieve_and_store_model(validator, hotkey: str):
    """Retrieve model metadata from chain and store in database."""
    try:
        uid = validator.metagraph.hotkeys.index(hotkey)
        chain_model_metadata = await validator.chain_models.retrieve_model_metadata(hotkey, uid)
    except Exception as e:
        bt.logging.warning(f"Cannot get miner model for {hotkey}: {e}")
        return None

    try:
        validator.db_controller.add_model(chain_model_metadata, hotkey)
        return chain_model_metadata
    except Exception as e:
        handle_model_storage_error(e, hotkey, chain_model_metadata)
        return None


def handle_model_storage_error(error: Exception, hotkey: str, metadata):
    """Handle errors when storing model metadata."""
    if "CHECK constraint failed: LENGTH(model_hash) <= 8" in str(error):
        bt.logging.error(
            f"Invalid model hash for {hotkey}: "
            f"Hash '{metadata.model_hash}' exceeds 8-character limit"
        )
    else:
        bt.logging.error(f"Failed to persist model info for {hotkey}: {error}", exc_info=True)


async def process_miner_models(validator, blacklisted_hotkeys: set):
    """Process each miner's model metadata."""
    for i, hotkey in enumerate(validator.hotkeys):
        if hotkey in blacklisted_hotkeys:
            bt.logging.debug(f"Skipping blacklisted hotkey {hotkey}")
            continue

        hotkey = str(hotkey)
        bt.logging.debug(f"Processing {i+1}/{len(validator.hotkeys)}: {hotkey}")
        
        await retrieve_and_store_model(validator, hotkey)


async def setup_organization_data_references(validator):
    """Setup organization data references by fetching, syncing, and returning instance."""
    from cancer_ai.validator.utils import fetch_organization_data_references, sync_organizations_data_references
    from cancer_ai.validator.utils import OrganizationDataReferenceFactory
    
    import bittensor as bt
    bt.logging.info(f"DEBUG: Fetching org data from {validator.config.datasets_config_hf_repo_id}")
    
    yaml_data = await fetch_organization_data_references(
        validator.config.datasets_config_hf_repo_id,
        validator.hf_api,
    )
    bt.logging.info(f"DEBUG: Fetched yaml_data: {yaml_data}")
    
    await sync_organizations_data_references(yaml_data)
    
    factory = OrganizationDataReferenceFactory.get_instance()
    bt.logging.info(f"DEBUG: Factory has {len(factory.organizations)} organizations")
    
    return factory
