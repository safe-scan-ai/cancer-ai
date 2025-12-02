"""
Structured logging system with context tracking for cancer-ai-subnet.
Provides categorized logging with competition and hotkey context.
"""

import logging
import threading
from typing import Optional, Dict
from enum import Enum
from datetime import datetime
from pathlib import Path


class LogLevel(Enum):
    ERROR = "ERROR"
    WARN = "WARN"
    WARNING = "WARNING"  # Python logging uses WARNING
    INFO = "INFO"
    DEBUG = "DEBUG"


class LogCategory(Enum):
    VALIDATION = "VALIDATION"
    BITTENSOR = "BITTENSOR"
    COMPETITION = "COMPETITION"
    INFERENCE = "INFERENCE"
    DATASET = "DATASET"
    STATISTICS = "STATISTICS"
    WANDB = "WANDB"
    CHAINSTORE = "CHAINSTORE"


class LogContext:
    """Thread-local logging context for competition and hotkey."""

    def __init__(self) -> None:
        self._local: threading.local = threading.local()

    def set_competition(self, competition_id: Optional[str]) -> None:
        self._local.competition_id = competition_id

    def set_hotkey(self, hotkey: Optional[str]) -> None:
        self._local.hotkey = hotkey

    def get_context(self) -> Dict[str, str]:
        return {
            "competition_id": getattr(self._local, "competition_id", None) or "",
            "hotkey": getattr(self._local, "hotkey", None) or "",
        }

    def clear(self) -> None:
        self._local.competition_id = None
        self._local.hotkey = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log output."""

    def __init__(self, context: LogContext) -> None:
        super().__init__()
        self._context = context

    def format(self, record: logging.LogRecord) -> str:
        # Get context information
        context = self._context.get_context()

        # Build the structured log line
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Handle both WARN and WARNING levels
        level_name = record.levelname
        if level_name == "WARNING":
            level_name = "WARN"
        level = level_name

        category = getattr(record, "category", LogCategory.VALIDATION.value)
        filename = f"{Path(record.pathname).name}:{record.lineno}"
        message = record.getMessage()

        # Single identifier: prefer hotkey, fall back to competition id
        identifier = context["hotkey"] or context["competition_id"] or ""

        return (
            f"{timestamp} | {level:5} | {category:11} | {filename} | "
            f"{identifier} | {message}"
        )


class StructuredLogger:
    """Main structured logger class."""

    def __init__(self, name: str = "cancer_ai") -> None:
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

            def error(self, message: str, **kwargs) -> None:
                self._parent.error(self._category, message, **kwargs)

            def warn(self, message: str, **kwargs) -> None:
                self._parent.warn(self._category, message, **kwargs)

            def info(self, message: str, **kwargs) -> None:
                self._parent.info(self._category, message, **kwargs)

            def debug(self, message: str, **kwargs) -> None:
                self._parent.debug(self._category, message, **kwargs)

        # Expose helpers as attributes
        self.validation = _CategoryLogger(self, LogCategory.VALIDATION)
        self.bittensor = _CategoryLogger(self, LogCategory.BITTENSOR)
        self.competition = _CategoryLogger(self, LogCategory.COMPETITION)
        self.inference = _CategoryLogger(self, LogCategory.INFERENCE)
        self.dataset = _CategoryLogger(self, LogCategory.DATASET)
        self.statistics = _CategoryLogger(self, LogCategory.STATISTICS)
        self.wandb = _CategoryLogger(self, LogCategory.WANDB)
        self.chainstore = _CategoryLogger(self, LogCategory.CHAINSTORE)
    
    def _log(self, level: LogLevel, category: LogCategory, message: str, **kwargs) -> None:
        """Internal logging method."""
        log_level = getattr(logging, level.name)
        extra = {"category": category.value}
        self.logger.log(log_level, message, extra=extra, **kwargs)

    def error(self, category: LogCategory, message: str, **kwargs) -> None:
        self._log(LogLevel.ERROR, category, message, **kwargs)

    def warn(self, category: LogCategory, message: str, **kwargs) -> None:
        self._log(LogLevel.WARN, category, message, **kwargs)

    def info(self, category: LogCategory, message: str, **kwargs) -> None:
        self._log(LogLevel.INFO, category, message, **kwargs)

    def debug(self, category: LogCategory, message: str, **kwargs) -> None:
        self._log(LogLevel.DEBUG, category, message, **kwargs)

    # Context management
    def set_competition(self, competition_id: str) -> None:
        self.context.set_competition(competition_id)

    def clear_competition(self) -> None:
        self.context.set_competition(None)

    def set_hotkey(self, hotkey: str) -> None:
        self.context.set_hotkey(hotkey)

    def clear_hotkey(self) -> None:
        self.context.set_hotkey(None)

    def clear_all_context(self) -> None:
        self.context.clear()


log = StructuredLogger()
