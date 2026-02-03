"""Axiom logging integration."""

import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

import bittensor as bt

import requests
from pydantic import BaseModel


class AxiomError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Axiom ingest failed: {status_code} {body}")
        self.status_code = status_code
        self.body = body


class ExcludeLoggerFilter(logging.Filter):
    """Exclude noisy/recursive loggers."""

    _ALLOWED_PREFIX: str = "bittensor"

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._ALLOWED_PREFIX)


class AxiomHandler(logging.Handler):
    """Send logs to Axiom as structured events."""

    class _AxiomEvent(BaseModel):
        _time: str
        level: str
        logger: str
        message: str
        file: str
        function: str
        validator: str
        validator_hotkey: str
        category: Optional[str] = None
        competition_id: Optional[str] = None
        competition_action: Optional[str] = None
        miner_hotkey: Optional[str] = None
        dataset: Optional[str] = None
        exception: Optional[str] = None

        class Config:
            extra = "allow"

    _RESERVED_RECORD_KEYS: set[str] = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "exc_info",
        "exc_text",
        "stack_info",
    }

    @staticmethod
    def _get_str(record: logging.LogRecord, key: str) -> Optional[str]:
        v = getattr(record, key, None)
        return v if isinstance(v, str) and v else None

    def __init__(
        self,
        session: requests.Session,
        ingest_base_url: str,
        dataset: str,
        *,
        validator_name: str,
        hotkey: str,
    ) -> None:
        super().__init__()
        self._session = session
        self._ingest_base_url = ingest_base_url.rstrip("/")
        self._dataset = dataset
        self._validator_name = validator_name
        self._hotkey = hotkey

    def send_startup_event(self) -> int:
        event_model = self._AxiomEvent(
            _time=datetime.now(timezone.utc).isoformat(),
            level="INFO",
            logger="bittensor.axiom",
            message="Axiom logging handshake",
            file="axiom_logging.py:0",
            function="send_startup_event",
            validator=self._validator_name,
            validator_hotkey=self._hotkey,
            category="AXIOM",
        )
        event: dict[str, Any] = event_model.model_dump(exclude_none=True)
        ingest_url = f"{self._ingest_base_url}/v1/ingest/{self._dataset}"
        resp = self._session.post(ingest_url, json=[event], timeout=2)
        if resp.status_code >= 400:
            raise AxiomError(resp.status_code, resp.text[:300])
        return resp.status_code

    def emit(self, record: logging.LogRecord) -> None:
        try:
            rel_path: str
            try:
                rel_path = os.path.relpath(record.pathname, start=os.getcwd())
            except ValueError:
                rel_path = record.pathname

            file_ref = f"{rel_path}:{record.lineno}"

            if record.exc_info:
                exception = self.format(record)
            else:
                exception = None

            event_model = self._AxiomEvent(
                _time=datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                level=record.levelname,
                logger=record.name,
                message=record.getMessage(),
                file=file_ref,
                function=record.funcName,
                validator=self._validator_name,
                validator_hotkey=self._hotkey,  # Should be full SS58 address
                category=self._get_str(record, "category"),
                competition_id=self._get_str(record, "competition_id"),
                competition_action=self._get_str(record, "competition_action"),
                miner_hotkey=self._get_str(record, "miner_hotkey"),
                dataset=self._get_str(record, "dataset"),
                exception=exception,
            )

            event: dict[str, Any] = event_model.model_dump(exclude_none=True)

            for k, v in record.__dict__.items():
                if k in self._RESERVED_RECORD_KEYS:
                    continue
                if k in event:
                    continue
                event[k] = v

            ingest_url = f"{self._ingest_base_url}/v1/ingest/{self._dataset}"
            resp = self._session.post(ingest_url, json=[event], timeout=2)
            if resp.status_code >= 400:
                raise AxiomError(resp.status_code, resp.text[:300])
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            if os.getenv("AXIOM_DEBUG") == "1":
                bt.logging.exception("Axiom emit failed")
            return


def setup_axiom_logging(config: "bt.Config") -> Optional[logging.Handler]:
    if not getattr(config, "logs_axiom_enabled", False):
        return None

    token = os.getenv("AXIOM_API_KEY")
    if not token:
        bt.logging.error("AXIOM_API_KEY is not set in environment")
        return None

    validator_name = getattr(config, "validator_name", None)
    if not validator_name:
        bt.logging.error("--validator_name is required when --logs_axiom_enabled is set")
        sys.exit(1)

    dataset = getattr(config, "axiom_dataset", "cancer-ai-logs")

    axiom_url = (getattr(config, "axiom_url", None) or os.getenv("AXIOM_URL"))
    if not axiom_url:
        bt.logging.error(
            "AXIOM_URL (or --axiom_url) must be set to your edge deployment base URL, e.g. https://eu-central-1.aws.edge.axiom.co"
        )
        return None

    hotkey_cfg = getattr(getattr(config, "wallet", None), "hotkey", "")
    if isinstance(hotkey_cfg, str):
        # Fallback: if it's a string, we can't get SS58 address without the wallet object
        hotkey = hotkey_cfg
    else:
        # Get the full SS58 address from the hotkey object
        hotkey = getattr(hotkey_cfg, "ss58_address", "")

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
    )
    org_id = os.getenv("AXIOM_ORG_ID")
    if org_id:
        session.headers.update({"X-Axiom-Org-Id": org_id})

    handler = AxiomHandler(
        session,
        axiom_url,
        dataset,
        validator_name=validator_name,
        hotkey=hotkey,
    )
    handler.addFilter(ExcludeLoggerFilter())

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    local_logger = logging.getLogger("axiom_startup")
    try:
        status_code = handler.send_startup_event()
        print(
            f"AXIOM_HANDSHAKE_OK dataset={dataset} validator={validator_name} url={axiom_url} status={status_code}",
            file=sys.stdout,
            flush=True,
        )
        local_logger.info(
            "Axiom logging handshake succeeded: dataset=%s validator=%s url=%s",
            dataset,
            validator_name,
            axiom_url,
        )
    except Exception as e:
        root_logger.removeHandler(handler)
        print(
            "AXIOM_HANDSHAKE_FAIL "
            f"dataset={dataset} validator={validator_name} url={axiom_url} error={e} "
            f"traceback={traceback.format_exc()}",
            file=sys.stdout,
            flush=True,
        )
        local_logger.error(
            "Axiom logging handshake FAILED: dataset=%s validator=%s url=%s error=%s",
            dataset,
            validator_name,
            axiom_url,
            str(e),
        )
        local_logger.exception("Axiom logging handshake exception")
        return None

    bt.logging.info(f"Axiom logging enabled: dataset={dataset} (validator={validator_name})")
    return handler
