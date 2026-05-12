from __future__ import annotations
import logging
import logging.config
from pathlib import Path
import yaml


DEFAULT_LOGGER_NAME = "persian_rag_temporal"


def setup_logger(config_path: str = "configs/logging.yaml") -> None:
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)
