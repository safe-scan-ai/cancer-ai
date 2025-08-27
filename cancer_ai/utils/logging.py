import os
import logging
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from pygelf import GelfUdpHandler

load_dotenv()

EVENTS_LEVEL_NUM = 38
DEFAULT_LOG_BACKUP_COUNT = 10


def setup_graylog(logger):
    if (
        not os.getenv("GRAYLOG_HOST")
        or not os.getenv("GRAYLOG_PORT")
        or not os.getenv("GRAYLOG_KEY")
        or not os.getenv("GRAYLOG_VALIDATOR_ID")
    ):
        # if we don't have env variables, don't setup graylog
        return
    logger.addHandler(
        GelfUdpHandler(
            host=os.getenv("GRAYLOG_HOST"),
            port=int(os.getenv("GRAYLOG_PORT")),
            _validator_hotkey=os.getenv("GRAYLOG_VALIDATOR_ID"),

        )
    )
    return logger


def setup_events_logger(full_path, events_retention_size):
    logging.addLevelName(EVENTS_LEVEL_NUM, "EVENT")

    logger = logging.getLogger("event")
    logger.setLevel(EVENTS_LEVEL_NUM)
    logger = setup_graylog(logger)
    def event(self, message, *args, **kws):
        if self.isEnabledFor(EVENTS_LEVEL_NUM):
            self._log(EVENTS_LEVEL_NUM, message, args, **kws)

    logging.Logger.event = event

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        os.path.join(full_path, "events.log"),
        maxBytes=events_retention_size,
        backupCount=DEFAULT_LOG_BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(EVENTS_LEVEL_NUM)
    logger.addHandler(file_handler)

    return logger
