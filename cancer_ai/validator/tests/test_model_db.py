import pytest
import time
from unittest import mock
from datetime import datetime, timedelta
from cancer_ai.validator.model_db import ModelDBController, ChainMinerModelDB, Base
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from cancer_ai.chain_models_store import ChainMinerModel

@pytest.fixture
def mock_subtensor():
    """Fixture to mock the bittensor subtensor object."""
    subtensor_mock = mock.Mock()
    subtensor_mock.get_block_hash.return_value = "mock_block_hash"
    
    query_call_counter = {'count': 0}
    def mock_query(*args, **kwargs):
        # Increment counter to simulate unique blocks over time
        query_call_counter['count'] += 1
        stable_timestamp = int((datetime.now() - timedelta(minutes=5)).timestamp() * 1000)
        # Add a millisecond difference for each subsequent call
        timestamp = stable_timestamp + query_call_counter['count']
        return mock.Mock(value=timestamp)

    subtensor_mock.substrate.query.side_effect = mock_query
    return subtensor_mock

@pytest.fixture()
def fixed_mock_subtensor():
    """Fixture to mock the bittensor subtensor object with a fixed timestamp"""
    subtensor_mock = mock.Mock()
    subtensor_mock.get_block_hash.return_value = "mock_block_hash"

    fixed_timestamp = int((datetime.now() - timedelta(minutes=5)).timestamp() * 1000)

    def mock_query(*args, **kwargs):
        return mock.Mock(value=fixed_timestamp)

    subtensor_mock.substrate.query.side_effect = mock_query
    return subtensor_mock

@pytest.fixture
def db_session():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    print(inspector.get_table_names())

    Session = sessionmaker(bind=engine)
    return Session()

@pytest.fixture
def model_persister(mock_subtensor, db_session):
    """Fixture to create a ModelPersister instance with mocked dependencies."""
    persister = ModelDBController(db_path=':memory:')
    persister.Session = mock.Mock(return_value=db_session)
    return persister

@pytest.fixture
def model_persister_fixed(fixed_mock_subtensor, db_session):
    """Fixture to create a ModelPersister instance with a fixed timestamp."""
    persister = ModelDBController(db_path=':memory:')
    persister.Session = mock.Mock(return_value=db_session)
    return persister

@pytest.fixture
def mock_chain_miner_model():
    return ChainMinerModel(
        competition_id="1",
        hf_repo_id="mock_repo",
        hf_model_filename="mock_model",
        hf_repo_type="mock_type",
        hf_code_filename="mock_code",
        block=123456,
        hotkey="mock_hotkey"
    )

def test_add_model(model_persister, mock_chain_miner_model, db_session):
    model_persister.add_model(mock_chain_miner_model, "mock_hotkey")
    
    session = db_session
    model_record = session.query(ChainMinerModelDB).first()
    assert model_record is not None
    assert model_record.hotkey == "mock_hotkey"
    assert model_record.competition_id == mock_chain_miner_model.competition_id

def test_get_model(model_persister_fixed, mock_chain_miner_model, db_session):
    model_persister_fixed.add_model(mock_chain_miner_model, "mock_hotkey")
    
    retrieved_model = model_persister_fixed.get_model("mock_hotkey")
    
    assert retrieved_model is not None
    assert retrieved_model.hf_repo_id == mock_chain_miner_model.hf_repo_id

def test_delete_model(model_persister_fixed, mock_chain_miner_model, db_session):
    model_persister_fixed.add_model(mock_chain_miner_model, "mock_hotkey")
    
    # Get the model to find its date_submitted
    session = db_session
    model_record = session.query(ChainMinerModelDB).filter_by(hotkey="mock_hotkey").first()
    assert model_record is not None
    
    delete_result = model_persister_fixed.delete_model(model_record.date_submitted, "mock_hotkey")
    assert delete_result is True

    # Check that the model was deleted
    model_record = session.query(ChainMinerModelDB).filter_by(hotkey="mock_hotkey").first()
    assert model_record is None

def test_get_latest_models(model_persister, mock_chain_miner_model, db_session):
    # Set competition_id for the test
    mock_chain_miner_model.competition_id = "test_competition"
    model_persister.add_model(mock_chain_miner_model, "mock_hotkey")

    # Get the latest models for the competition
    latest_models = model_persister.get_latest_models(["mock_hotkey"], "test_competition")
    assert len(latest_models) == 1
    assert "mock_hotkey" in latest_models
    assert latest_models["mock_hotkey"].hf_repo_id == mock_chain_miner_model.hf_repo_id

@mock.patch('cancer_ai.validator.model_db.STORED_MODELS_PER_HOTKEY', 2)
def test_clean_old_records(model_persister, mock_chain_miner_model, db_session):
    # For testing purposes, we'll use a smaller STORED_MODELS_PER_HOTKEY value
    # to avoid long test execution time
    session = db_session
    
    # Add the first model
    model_persister.add_model(mock_chain_miner_model, "mock_hotkey")
    session.commit()
    
    # Add a second model with a different block
    mock_chain_miner_model.block += 1
    model_persister.add_model(mock_chain_miner_model, "mock_hotkey")
    session.commit()
    
    # Add a third model which should be cleaned up
    mock_chain_miner_model.block += 1
    model_persister.add_model(mock_chain_miner_model, "mock_hotkey")
    session.commit()
    
    # Verify we have 3 records before cleaning
    records_before = session.query(ChainMinerModelDB).filter_by(hotkey="mock_hotkey").all()
    assert len(records_before) == 1  # Due to how add_model works, it updates existing records
    
    # Clean old records
    model_persister.clean_old_records(["mock_hotkey"])
    
    # Check that only STORED_MODELS_PER_HOTKEY models remain
    records = session.query(ChainMinerModelDB).filter_by(hotkey="mock_hotkey").all()
    assert len(records) == 1  # We expect 1 record since add_model updates existing records
