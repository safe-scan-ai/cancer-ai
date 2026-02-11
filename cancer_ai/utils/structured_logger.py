"""
Structured logging system with context tracking for cancer-ai-subnet.
Provides categorized logging with competition and hotkey context.
"""

import logging
import threading
from typing import Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
from pathlib import Path


class LogLevel(Enum):
    ERROR = "ERROR"
    WARN = "WARN"
    WARNING = "WARNING"  # Python logging uses WARNING
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"


class LogCategory(Enum):
    VALIDATION = "VALIDATION"
    BITTENSOR = "BITTENSOR"
    COMPETITION = "COMPETITION"
    INFERENCE = "INFERENCE"
    DATASET = "DATASET"
    STATISTICS = "STATISTICS"
    WANDB = "WANDB"
    CHAINSTORE = "CHAINSTORE"
    COMMUNICATION = "COMMUNICATION"
    STATE_SYNC = "STATE_SYNC"
    INTERNAL = "INTERNAL"


class LogContext:
    """Thread-local logging context for competition and hotkey."""

    def __init__(self) -> None:
        self._local = threading.local()

    def set_competition(self, competition_id: Optional[str]) -> None:
        self._local.competition_id = competition_id

    def set_competition_action(self, action: Optional[str]) -> None:
        self._local.competition_action = action


    def set_miner_hotkey(self, hotkey: Optional[str]) -> None:
        self._local.miner_hotkey = hotkey

    def set_dataset(self, dataset_hf_repo: Optional[str], dataset_hf_filename: Optional[str]) -> None:
        if dataset_hf_repo and dataset_hf_filename:
            self._local.dataset = f"{dataset_hf_repo}/{dataset_hf_filename}"
        else:
            self._local.dataset = None

    def get_context(self) -> Dict[str, str]:
        return {
            "competition_id": getattr(self._local, "competition_id", None) or "",
            "competition_action": getattr(self._local, "competition_action", None) or "",
            "miner_hotkey": getattr(self._local, "miner_hotkey", None) or "",
            "dataset": getattr(self._local, "dataset", None) or "",
        }

    def clear(self) -> None:
        self._local.competition_id = None
        self._local.competition_action = None
        self._local.validator_hotkey = None
        self._local.miner_hotkey = None
        self._local.dataset = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log output."""

    def __init__(self, context: LogContext) -> None:
        super().__init__()
        self._context = context

    def format(self, record: logging.LogRecord) -> str:
        # Build the structured log line
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Handle both WARN and WARNING levels, and show TRACE properly
        level_name = record.levelname
        if level_name == "WARNING":
            level_name = "WARN"
        elif level_name == "DEBUG" and getattr(record, 'is_trace', False):
            level_name = "TRACE"
        level = level_name

        category = getattr(record, "category", LogCategory.VALIDATION.value)
        message = record.getMessage()

        miner_hotkey = getattr(record, "miner_hotkey", "")

        return (
            f"{timestamp} | {level:4} | {category:11} | "
            f"{miner_hotkey} | {message}"
        )


class StructuredLogger:
    """Main structured logger class."""

    def __init__(self, name: str = "bittensor.structured") -> None:
        self.logger = logging.getLogger(name)
        self.context = LogContext()
        self._setup_logger()
        self._init_category_helpers()

    def _setup_logger(self) -> None:
        """Setup the structured logger with custom formatter."""
        if not self.logger.handlers:
            formatter = StructuredFormatter(self.context)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File handler
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "structured.log")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.logger.setLevel(logging.DEBUG)

    def _init_category_helpers(self) -> None:
        """Create per-category helper proxies (e.g. log.validation.info)."""

        class _CategoryLogger:
            def __init__(self, parent: "StructuredLogger", category: LogCategory) -> None:
                self._parent = parent
                self._category = category

            def error(self, message: str, **kwargs: Any) -> None:
                self._parent.error(self._category, message, **kwargs)

            def warn(self, message: str, **kwargs: Any) -> None:
                self._parent.warn(self._category, message, **kwargs)

            def warning(self, message: str, **kwargs: Any) -> None:
                self._parent.warn(self._category, message, **kwargs)

            def info(self, message: str, **kwargs: Any) -> None:
                self._parent.info(self._category, message, **kwargs)

            def debug(self, message: str, **kwargs: Any) -> None:
                self._parent.debug(self._category, message, **kwargs)

            def trace(self, message: str, **kwargs: Any) -> None:
                self._parent.trace(self._category, message, **kwargs)
                
            def exception(self, message: str, **kwargs: Any) -> None:
                self._parent.exception(self._category, message, **kwargs)

        self.validation = _CategoryLogger(self, LogCategory.VALIDATION)
        self.bittensor = _CategoryLogger(self, LogCategory.BITTENSOR)
        self.competition = _CategoryLogger(self, LogCategory.COMPETITION)
        self.inference = _CategoryLogger(self, LogCategory.INFERENCE)
        self.dataset = _CategoryLogger(self, LogCategory.DATASET)
        self.statistics = _CategoryLogger(self, LogCategory.STATISTICS)
        self.wandb = _CategoryLogger(self, LogCategory.WANDB)
        self.chainstore = _CategoryLogger(self, LogCategory.CHAINSTORE)
        self.communication = _CategoryLogger(self, LogCategory.COMMUNICATION)
        self.state_sync = _CategoryLogger(self, LogCategory.STATE_SYNC)
        self.internal = _CategoryLogger(self, LogCategory.INTERNAL)

    def _log(self, level: LogLevel, category: LogCategory, message: str, **kwargs: Any) -> None:
        """Internal logging method."""
        # Python logging doesn’t have TRACE level, so map it to DEBUG
        if level == LogLevel.TRACE:
            log_level = logging.DEBUG
            is_trace = True
        else:
            log_level = getattr(logging, level.name)
            is_trace = False
        
        extra: dict[str, Any] = {"category": category.value}
        if is_trace:
            extra["is_trace"] = True

        context = self.context.get_context()
        if context["competition_id"]:
            extra["competition_id"] = context["competition_id"]
        if context["competition_action"]:
            extra["competition_action"] = context["competition_action"]
        if context["miner_hotkey"]:
            extra["miner_hotkey"] = context["miner_hotkey"]

        user_extra = kwargs.pop("extra", None)
        if isinstance(user_extra, dict):
            extra.update(user_extra)

        self.logger.log(log_level, message, extra=extra, stacklevel=3, **kwargs)

    # Logging methods with optional category parameter
    def error(self, message_or_category: Union[str, LogCategory], message: Optional[str] = None, **kwargs: Any) -> None:
        if isinstance(message_or_category, LogCategory) and message is not None:
            # Category-specific logging
            self._log(LogLevel.ERROR, message_or_category, message, **kwargs)
        else:
            # Default category logging
            self._log(LogLevel.ERROR, LogCategory.VALIDATION, message_or_category, **kwargs)
        
    def warn(self, message_or_category: Union[str, LogCategory], message: Optional[str] = None, **kwargs: Any) -> None:
        if isinstance(message_or_category, LogCategory) and message is not None:
            self._log(LogLevel.WARN, message_or_category, message, **kwargs)
        else:
            self._log(LogLevel.WARN, LogCategory.VALIDATION, message_or_category, **kwargs)

    def warning(self, message_or_category: Union[str, LogCategory], message: Optional[str] = None, **kwargs: Any) -> None:
        self.warn(message_or_category, message, **kwargs)

    def info(self, message_or_category: Union[str, LogCategory], message: Optional[str] = None, **kwargs: Any) -> None:
        if isinstance(message_or_category, LogCategory) and message is not None:
            self._log(LogLevel.INFO, message_or_category, message, **kwargs)
        else:
            self._log(LogLevel.INFO, LogCategory.VALIDATION, message_or_category, **kwargs)

    def trace(self, message_or_category: Union[str, LogCategory], message: Optional[str] = None, **kwargs: Any) -> None:
        if isinstance(message_or_category, LogCategory) and message is not None:
            self._log(LogLevel.TRACE, message_or_category, message, **kwargs)
        else:
            self._log(LogLevel.TRACE, LogCategory.VALIDATION, message_or_category, **kwargs)

    def debug(self, message_or_category: Union[str, LogCategory], message: Optional[str] = None, **kwargs: Any) -> None:
        if isinstance(message_or_category, LogCategory) and message is not None:
            self._log(LogLevel.DEBUG, message_or_category, message, **kwargs)
        else:
            self._log(LogLevel.DEBUG, LogCategory.VALIDATION, message_or_category, **kwargs)

    def exception(self, message_or_category: Union[str, LogCategory], message: Optional[str] = None, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        if isinstance(message_or_category, LogCategory) and message is not None:
            self._log(LogLevel.ERROR, message_or_category, message, **kwargs)
        else:
            self._log(LogLevel.ERROR, LogCategory.VALIDATION, message_or_category, **kwargs)

    def set_competition(self, competition_id: str) -> None:
        self.context.set_competition(competition_id)

    def set_competition_action(self, action: str) -> None:
        self.context.set_competition_action(action)

    def set_miner_hotkey(self, hotkey: str) -> None:
        self.context.set_miner_hotkey(hotkey)

    def set_hotkey(self, hotkey: str) -> None:
        self.context.set_hotkey(hotkey)

    def set_dataset(self, dataset_hf_repo: str, dataset_hf_filename: str) -> None:
        self.context.set_dataset(dataset_hf_repo, dataset_hf_filename)

    def clear_competition(self) -> None:
        self.context.set_competition(None)

    def clear_miner_hotkey(self) -> None:
        self.context.set_miner_hotkey(None)

    def clear_all_context(self) -> None:
        self.context.clear()

    def install_bittensor_logger_bridge(self) -> None:
        bt_logger = logging.getLogger("bittensor")
        for flt in bt_logger.filters:
            if isinstance(flt, _ContextInjectionFilter):
                return
        bt_logger.addFilter(_ContextInjectionFilter(self.context))


class _ContextInjectionFilter(logging.Filter):
    def __init__(self, context: LogContext) -> None:
        super().__init__()
        self._context = context

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = self._context.get_context()
        if ctx["competition_id"] and not hasattr(record, "competition_id"):
            record.competition_id = ctx["competition_id"]
        if ctx["competition_action"] and not hasattr(record, "competition_action"):
            record.competition_action = ctx["competition_action"]
        if ctx["miner_hotkey"] and not hasattr(record, "miner_hotkey"):
            record.miner_hotkey = ctx["miner_hotkey"]
        return True


log = StructuredLogger()