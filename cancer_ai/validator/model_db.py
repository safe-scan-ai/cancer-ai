import bittensor as bt
import os
from sqlalchemy import create_engine, Column, String, DateTime, PrimaryKeyConstraint, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta, timezone
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
    def __init__(self, db_path: str = "models.db"):
        db_url = f"sqlite:///{os.path.abspath(db_path)}"
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
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
        bt.logging.debug(f"Getting latest DB model for hotkey {hotkey}")
        session = self.Session()
        try:
            try:
                model_record = (
                    session.query(ChainMinerModelDB)
                    .filter(ChainMinerModelDB.hotkey == hotkey)
                    .order_by(ChainMinerModelDB.date_submitted.desc())
                    .first()
                )
                if model_record:
                    return self.convert_db_model_to_chain_model(model_record)
                return None
            except Exception as e:
                import traceback
                stack_trace = traceback.format_exc()
                bt.logging.error(f"Error in get_latest_model for hotkey {hotkey}: {e}")
                bt.logging.error(f"Stack trace: {stack_trace}")
                # Re-raise the exception to be caught by higher-level error handlers
                raise
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
            existing_model = session.query(ChainMinerModelDB).filter_by(
                hotkey=hotkey
            ).first()
            
            if existing_model:
                existing_model.competition_id = chain_miner_model.competition_id
                existing_model.hf_repo_id = chain_miner_model.hf_repo_id
                existing_model.hf_model_filename = chain_miner_model.hf_model_filename
                existing_model.hf_repo_type = chain_miner_model.hf_repo_type
                existing_model.hf_code_filename = chain_miner_model.hf_code_filename

                session.commit()
                bt.logging.debug(f"Successfully updated DB model for hotkey {hotkey}.")
                return True
            else:
                bt.logging.debug(f"No existing DB model found for hotkey {hotkey}. Update skipped.")
                return False

        except Exception as e:
            session.rollback()
            bt.logging.error(f"Error updating DB model for hotkey {hotkey}: {e}")
            raise e
        finally:
            session.close()


    def get_latest_models(self, hotkeys: list[str], competition_id: str) -> dict[str, ChainMinerModel]:
        # cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=cutoff_time) if cutoff_time else datetime.now(timezone.utc)
        session = self.Session()
        try:
            # Use a correlated subquery to get the latest record for each hotkey that doesn't violate the cutoff
            latest_models_to_hotkeys = {}
            for hotkey in hotkeys:
                model_record = (
                    session.query(ChainMinerModelDB)
                    .filter(ChainMinerModelDB.hotkey == hotkey)
                    .filter(ChainMinerModelDB.competition_id == competition_id)
                    # .filter(ChainMinerModelDB.date_submitted < cutoff_time)
                    # .order_by(ChainMinerModelDB.date_submitted.desc())  # Order by newest first
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
            date_submitted = datetime.now(timezone.utc), # temporary fix, can't be null
            block = 1, # temporary fix , can't be null
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