import bittensor as bt
import os
from sqlalchemy import create_engine, Column, String, DateTime, PrimaryKeyConstraint, Integer, func, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from datetime import datetime, timedelta
from ..chain_models_store import ChainMinerModel

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

    __table_args__ = (
        PrimaryKeyConstraint('date_submitted', 'hotkey', name='pk_date_hotkey'),
    )

class ModelDBController:
    def __init__(self, subtensor: bt.subtensor, db_path: str = "models.db"):
        self.subtensor = subtensor
        db_url = f"sqlite:///{os.path.abspath(db_path)}"
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def add_model(self, chain_miner_model: ChainMinerModel, hotkey: str):
        session = self.Session()
        date_submitted = self.get_block_timestamp(chain_miner_model.block)
        existing_model = self.get_model(date_submitted, hotkey)
        if not existing_model:
            try:
                model_record = self.convert_chain_model_to_db_model(chain_miner_model, hotkey)
                session.add(model_record)
                session.commit()
                bt.logging.debug(f"Successfully added model info for hotkey {hotkey} into the DB.")
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
        else:
            bt.logging.debug(f"Model for hotkey {hotkey} and date {date_submitted} already exists, proceeding with updating the model info.")
            self.update_model(chain_miner_model, hotkey)

    def get_model(self, date_submitted: datetime, hotkey: str):
        session = self.Session()
        try:
            model_record = session.query(ChainMinerModelDB).filter_by(
                date_submitted=date_submitted, hotkey=hotkey
            ).first()
            if model_record:
                return self.convert_db_model_to_chain_model(model_record)
            return None
        finally:
            session.close()

    def get_latest_model(self, hotkey: str, cutoff_time: float = None) -> ChainMinerModel:
        bt.logging.debug(f"Getting latest model for hotkey {hotkey} with cutoff time {cutoff_time}")
        cutoff_time = datetime.now() - timedelta(minutes=cutoff_time) if cutoff_time else datetime.now()
        session = self.Session()
        try:
            model_record = (
                session.query(ChainMinerModelDB)
                .filter(ChainMinerModelDB.hotkey == hotkey)
                .filter(ChainMinerModelDB.date_submitted < cutoff_time)
                .order_by(ChainMinerModelDB.date_submitted.desc())
                .first()
            )
            if model_record:
                return self.convert_db_model_to_chain_model(model_record)
            return None
        finally:
            session.close()

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
            date_submitted = self.get_block_timestamp(chain_miner_model.block)
            existing_model = session.query(ChainMinerModelDB).filter_by(
                date_submitted=date_submitted,
                hotkey=hotkey
            ).first()
            
            if existing_model:
                existing_model.competition_id = chain_miner_model.competition_id
                existing_model.hf_repo_id = chain_miner_model.hf_repo_id
                existing_model.hf_model_filename = chain_miner_model.hf_model_filename
                existing_model.hf_repo_type = chain_miner_model.hf_repo_type
                existing_model.hf_code_filename = chain_miner_model.hf_code_filename
                existing_model.block = chain_miner_model.block
                existing_model.date_submitted = date_submitted

                session.commit()
                bt.logging.debug(f"Successfully updated model for hotkey {hotkey} and date {date_submitted}.")
                return True
            else:
                bt.logging.debug(f"No existing model found for hotkey {hotkey} and date {date_submitted}. Update skipped.")
                return False

        except Exception as e:
            session.rollback()
            bt.logging.error(f"Error updating model for hotkey {hotkey} and date {date_submitted}: {e}")
            raise e
        finally:
            session.close()

    def get_block_timestamp(self, block_number):
        """Gets the timestamp of a block given its number."""
        try:
            block_hash = self.subtensor.get_block_hash(block_number)
            if block_hash is None:
                raise ValueError(f"Block hash not found for block number {block_number}")
            
            timestamp_info = self.subtensor.substrate.query(
                module='Timestamp',
                storage_function='Now',
                block_hash=block_hash
            )

            if timestamp_info is None:
                raise ValueError(f"Timestamp not found for block hash {block_hash}")

            timestamp_ms = timestamp_info.value
            block_datetime = datetime.fromtimestamp(timestamp_ms / 1000.0)

            return block_datetime
        except Exception as e:
            bt.logging.error(f"Error retrieving block timestamp: {e}")
            raise


    def get_all_models_for_competition(self, competition_id: str, cutoff_time: float | None = None) -> list[tuple[str,ChainMinerModel]]:
        cutoff_time = datetime.now() - timedelta(minutes=cutoff_time) if cutoff_time else datetime.now()
        session = self.Session()
        try:
            # Subquery to get the latest submission date for each hotkey
            latest_dates = (
                session.query(
                    ChainMinerModelDB.hotkey,
                    func.max(ChainMinerModelDB.date_submitted).label('max_date')
                )
                .filter(ChainMinerModelDB.competition_id == competition_id)
                .filter(ChainMinerModelDB.date_submitted < cutoff_time)
                .group_by(ChainMinerModelDB.hotkey)
                .subquery()
            )

            # Main query joining with the subquery to get the latest model for each hotkey
            models = (
                session.query(ChainMinerModelDB)
                .join(
                    latest_dates,
                    and_(
                        ChainMinerModelDB.hotkey == latest_dates.c.hotkey,
                        ChainMinerModelDB.date_submitted == latest_dates.c.max_date
                    )
                )
                .all()
            )
            return [(model.hotkey, self.convert_db_model_to_chain_model(model)) for model in models]
        finally:
            session.close()

    def get_latest_models(self, hotkeys: list[str], cutoff_time: float = None) -> dict[str, ChainMinerModel]:
        cutoff_time = datetime.now() - timedelta(minutes=cutoff_time) if cutoff_time else datetime.now()
        session = self.Session()
        try:
            # Use a correlated subquery to get the latest record for each hotkey that doesn't violate the cutoff
            latest_models_to_hotkeys = {}
            for hotkey in hotkeys:
                model_record = (
                    session.query(ChainMinerModelDB)
                    .filter(ChainMinerModelDB.hotkey == hotkey)
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
            bt.logging.error(f"Error deleting records for hotkeys not in list: {e}")

        finally:
            session.close()

    def convert_chain_model_to_db_model(self, chain_miner_model: ChainMinerModel, hotkey: str) -> ChainMinerModelDB:
        date_submitted = self.get_block_timestamp(chain_miner_model.block)
        return ChainMinerModelDB(
            competition_id = chain_miner_model.competition_id,
            hf_repo_id = chain_miner_model.hf_repo_id,
            hf_model_filename = chain_miner_model.hf_model_filename,
            hf_repo_type = chain_miner_model.hf_repo_type,
            hf_code_filename = chain_miner_model.hf_code_filename,
            date_submitted = date_submitted,
            block = chain_miner_model.block,
            hotkey = hotkey
        )

    def convert_db_model_to_chain_model(self, model_record: ChainMinerModelDB) -> ChainMinerModel:
        return ChainMinerModel(
            competition_id=model_record.competition_id,
            hf_repo_id=model_record.hf_repo_id,
            hf_model_filename=model_record.hf_model_filename,
            hf_repo_type=model_record.hf_repo_type,
            hf_code_filename=model_record.hf_code_filename,
            block=model_record.block,
        )