import bittensor as bt
import os
import re, traceback

import traceback
from sqlalchemy import create_engine, Column, String, DateTime, PrimaryKeyConstraint, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta, timezone
from ..chain_models_store import ChainMinerModel
from websockets.client import OPEN as WS_OPEN

from retry import retry

Base = declarative_base()

STORED_MODELS_PER_HOTKEY = 10

class ChainMinerModelDB(Base):
    __tablename__ = 'models'
    competition_id = Column(String, nullable=False)
    hf_repo_id = Column(String, nullable=False)
    hf_model_filename = Column(String, nullable=False)
    hf_repo_type = Column(String, nullable=False)
    hf_code_filename = Column(String, nullable=False)
    date_submitted = Column(DateTime, nullable=False)
    block = Column(Integer, nullable=False)
    hotkey = Column(String, nullable=False)
    model_hash = Column(String, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('date_submitted', 'hotkey', name='pk_date_hotkey'),
    )

class ModelDBController:
    def __init__(self, db_path: str = "models.db", subtensor: bt.subtensor = None):
        db_url = f"sqlite:///{os.path.abspath(db_path)}"
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        if subtensor is not None and "test" not in subtensor.chain_endpoint.lower():
            subtensor = bt.subtensor(network="archive")
        self.subtensor = subtensor

        # Capture the original connect() and override with _ws_connect wrapper
        # Substrate-interface calls connect() on every RPC under the hood,
        # so we wrap it to reuse the same socket unless it's truly closed.
        self._orig_ws_connect = self.subtensor.substrate.connect
        self.subtensor.substrate.connect = self._ws_connect

        ws = self.subtensor.substrate.connect()
        bt.logging.info(f"Initial WebSocket state: {ws.state}")

        self._migrate_database()

    def _ws_connect(self, *args, **kwargs):
        """
        Replacement for substrate.connect().
        Reuses existing WebSocketClientProtocol if State.OPEN;
        otherwise performs a fresh handshake via original connect().
        """
        # Check current socket
        current = getattr(self.subtensor.substrate, "ws", None)
        if current is not None and current.state == WS_OPEN:
            return current

        # If socket not open, reconnect
        bt.logging.warning("⚠️ Subtensor WebSocket not OPEN—reconnecting…")
        try:
            new_ws = self._orig_ws_connect(*args, **kwargs)
        except Exception as e:
            bt.logging.error("Failed to reconnect WebSocket: %s", e, exc_info=True)
            raise

        # Update the substrate.ws attribute so future calls reuse this socket
        setattr(self.subtensor.substrate, "ws", new_ws)
        return new_ws

    def _migrate_database(self):
        """Check and apply migration for model_hash column if missing."""
        with self.engine.connect() as connection:
            result = connection.execute("PRAGMA table_info(models)").fetchall()
            column_names = [row[1] for row in result]
            if "model_hash" not in column_names:
                try:
                    connection.execute("ALTER TABLE models ADD COLUMN model_hash TEXT CHECK(LENGTH(model_hash) <= 8)")
                    bt.logging.info("Migrated database: Added model_hash column with length constraint to models table")
                except Exception as e:
                    bt.logging.error(f"Failed to migrate database: {e}")
                    raise


    
    def add_model(self, chain_miner_model: ChainMinerModel, hotkey: str):
        session = self.Session()
        existing_model = self.get_model(hotkey)
        if not existing_model:
            try:
                model_record = self.convert_chain_model_to_db_model(chain_miner_model, hotkey)
                session.add(model_record)
                session.commit()
                bt.logging.debug(f"Successfully added DB model info for hotkey {hotkey} into the DB.")
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
        else:
            bt.logging.debug(f"DB model for hotkey {hotkey} already exists, proceeding with updating the model info.")
            self.update_model(chain_miner_model, hotkey)

    def get_model(self, hotkey: str) -> ChainMinerModel | None:
        session = self.Session()
        try:
            model_record = session.query(ChainMinerModelDB).filter_by(
                hotkey=hotkey
            ).first()
            if model_record:
                return self.convert_db_model_to_chain_model(model_record)
            return None
        finally:
            session.close()

    def get_latest_model(self, hotkey: str, cutoff_time: float = None) -> ChainMinerModel | None:
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=cutoff_time) if cutoff_time else datetime.now(timezone.utc)
        bt.logging.trace(f"Getting latest DB model for hotkey {hotkey}")
        session = self.Session()
        model_record = None
        try:
            model_record = (
                session.query(ChainMinerModelDB)
                .filter(ChainMinerModelDB.hotkey == hotkey)
                .filter(ChainMinerModelDB.date_submitted < cutoff_time)
                .order_by(ChainMinerModelDB.date_submitted.desc())
                .first()
            )
        except Exception as e:
            bt.logging.error(f"Error in get_latest_model for hotkey {hotkey}: {e}\n {traceback.format_exc()}")
            raise
        finally:
            session.close()

        if not model_record:
            return None
        
        return self.convert_db_model_to_chain_model(model_record)

    def delete_model(self, date_submitted: datetime, hotkey: str):
        session = self.Session()
        try:
            model_record = session.query(ChainMinerModelDB).filter_by(
                date_submitted=date_submitted, hotkey=hotkey
            ).first()
            if model_record:
                session.delete(model_record)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def update_model(self, chain_miner_model: ChainMinerModel, hotkey: str):
        session = self.Session()
        try:
            existing_model = session.query(ChainMinerModelDB).filter_by(
                hotkey=hotkey
            ).first()
            
            if existing_model:
                existing_model.competition_id = chain_miner_model.competition_id
                existing_model.hf_repo_id = chain_miner_model.hf_repo_id
                existing_model.hf_model_filename = chain_miner_model.hf_model_filename
                existing_model.hf_repo_type = chain_miner_model.hf_repo_type
                existing_model.hf_code_filename = chain_miner_model.hf_code_filename
                existing_model.date_submitted = self.get_block_timestamp(chain_miner_model.block)
                existing_model.block = chain_miner_model.block
                existing_model.model_hash = chain_miner_model.model_hash

                session.commit()
                bt.logging.debug(f"Successfully updated DB model for hotkey {hotkey}.")
                return True
            else:
                bt.logging.debug(f"No existing DB model found for hotkey {hotkey}. Update skipped.")
                return False

        except Exception as e:
            session.rollback()
            bt.logging.error(f"Error updating DB model for hotkey {hotkey}: {e}", exc_info=True)
            raise e
        finally:
            session.close()


    def get_latest_models(self, hotkeys: list[str], competition_id: str, cutoff: int = None) -> dict[str, ChainMinerModel]:
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=cutoff) if cutoff else datetime.now(timezone.utc)
        session = self.Session()
        try:
            # Use a correlated subquery to get the latest record for each hotkey that doesn't violate the cutoff
            latest_models_to_hotkeys = {}
            for hotkey in hotkeys:
                model_record = (
                    session.query(ChainMinerModelDB)
                    .filter(ChainMinerModelDB.hotkey == hotkey)
                    .filter(ChainMinerModelDB.competition_id == competition_id)
                    .filter(ChainMinerModelDB.date_submitted < cutoff_time)
                    .order_by(ChainMinerModelDB.date_submitted.desc())  # Order by newest first
                    .first()  # Get the first (newest) record that meets the cutoff condition
                )
                if model_record:
                    latest_models_to_hotkeys[hotkey] = self.convert_db_model_to_chain_model(model_record)

            return latest_models_to_hotkeys
        finally:
            session.close()

    def clean_old_records(self, hotkeys: list[str]):
        session = self.Session()

        for hotkey in hotkeys:
            try:
                records = (
                    session.query(ChainMinerModelDB)
                    .filter(ChainMinerModelDB.hotkey == hotkey)
                    .order_by(ChainMinerModelDB.date_submitted.desc())
                    .all()
                )

                # If there are more than STORED_MODELS_PER_HOTKEY records, delete the oldest ones
                if len(records) > STORED_MODELS_PER_HOTKEY:
                    records_to_delete = records[STORED_MODELS_PER_HOTKEY:]
                    for record in records_to_delete:
                        session.delete(record)

                session.commit()

            except Exception as e:
                session.rollback()
                bt.logging.error(f"Error processing hotkey {hotkey}: {e}")

        try:
            # Delete all records for hotkeys not in the given list
            session.query(ChainMinerModelDB).filter(ChainMinerModelDB.hotkey.notin_(hotkeys)).delete(synchronize_session=False)
            session.commit()
        except Exception as e:
            session.rollback()
            bt.logging.error(f"Error deleting DB records for hotkeys not in list: {e}")

        finally:
            session.close()

    def convert_chain_model_to_db_model(self, chain_miner_model: ChainMinerModel, hotkey: str) -> ChainMinerModelDB:
        return ChainMinerModelDB(
            competition_id = chain_miner_model.competition_id,
            hf_repo_id = chain_miner_model.hf_repo_id,
            hf_model_filename = chain_miner_model.hf_model_filename,
            hf_repo_type = chain_miner_model.hf_repo_type,
            hf_code_filename = chain_miner_model.hf_code_filename,
            date_submitted = self.get_block_timestamp(chain_miner_model.block),
            block = chain_miner_model.block,
            hotkey = hotkey,
            model_hash=chain_miner_model.model_hash
        )

    def convert_db_model_to_chain_model(self, model_record: ChainMinerModelDB) -> ChainMinerModel:
        return ChainMinerModel(
            competition_id=model_record.competition_id,
            hf_repo_id=model_record.hf_repo_id,
            hf_model_filename=model_record.hf_model_filename,
            hf_repo_type=model_record.hf_repo_type,
            hf_code_filename=model_record.hf_code_filename,
            block=model_record.block,
            model_hash=model_record.model_hash,
        )
    
    def compare_hotkeys(
        self, hotkey1: str, hotkey2: str
    ) -> tuple[str | None, datetime | None]:
        """
        Compares two hotkeys in the DB and returns (earliest_hotkey, earliest_date_submitted).
        If neither hotkey has any record, returns (None, None).
        If only one hotkey has a record, that one is automatically considered 'earlier'.
        """
        session = self.Session()
        try:
            record1 = (
                session.query(ChainMinerModelDB)
                .filter(ChainMinerModelDB.hotkey == hotkey1)
                .order_by(ChainMinerModelDB.date_submitted.asc())
                .first()
            )

            record2 = (
                session.query(ChainMinerModelDB)
                .filter(ChainMinerModelDB.hotkey == hotkey2)
                .order_by(ChainMinerModelDB.date_submitted.asc())
                .first()
            )

            if record1 is None and record2 is None:
                bt.logging.info(
                    f"No records found for either hotkey: {hotkey1} or {hotkey2}"
                )
                return None, None

            if record1 is None:
                bt.logging.info(
                    f"No DB record for hotkey {hotkey1}, so {hotkey2} is automatically earlier."
                )
                return hotkey2, record2.date_submitted

            if record2 is None:
                bt.logging.info(
                    f"No DB record for hotkey {hotkey2}, so {hotkey1} is automatically earlier."
                )
                return hotkey1, record1.date_submitted

            if record1.date_submitted <= record2.date_submitted:
                bt.logging.info(
                    f"hotkey {hotkey1} chosen as pioneer hotkey"
                )
                return hotkey1, record1.date_submitted
            else:
                bt.logging.info(
                    f"hotkey {hotkey2} chosen as pioneer hotkey"
                )
                return hotkey2, record2.date_submitted

        except Exception as e:
            session.rollback()
            bt.logging.error(f"Error comparing hotkeys {hotkey1} & {hotkey2}: {e}")
            raise
        finally:
            session.close()

    @retry(tries=10, delay=1, backoff=2, max_delay=30)
    def get_block_timestamp(self, block_number) -> datetime:
        """Gets the timestamp of a given block."""
        try:
            block_hash = self.subtensor.get_block_hash(block_number)

            if block_hash is None:
                raise ValueError(f"Block hash not found for block number {block_number}")

            timestamp_info = self.subtensor.substrate.query(
                module="Timestamp",
                storage_function="Now",
                block_hash=block_hash
            )

            if timestamp_info is None:
                raise ValueError(f"Timestamp not found for block hash {block_hash}")

            timestamp_ms = timestamp_info.value
            block_datetime = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)

            return block_datetime
        except Exception as e:
            bt.logging.exception(f"Error retrieving block timestamp: {e}")
            raise

    def close(self):
        try:
            bt.logging.debug("Closing ModelDBController and websocket connection.")
            self.subtensor.substrate.close_websocket()
        except Exception:
            pass
