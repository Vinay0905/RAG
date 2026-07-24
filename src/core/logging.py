import json 
import logging 
import sys 
from datetime import datetime
from config.settings import settings

class StructuredJSONFormatter(logging.Formatter):
    """Formate log record outputs into structured JSON strings"""
    def format(self,record :logging.LogRecord)->str:
        log_data={
            "timestamp":datetime.now().isoformat()+"Z",
            "level":record.levelname,
            "logger":record.getMessage(),
            "module":record.module,
            "filename":record.filename,
            "line_no":record.lineno,
        }
        if hasattr(record,"extra")and isinstance(record.extra,dict):
            log_data["extra"]=record.extra

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)
    def setup_logger(name: str = "cuad_rag") -> logging.Logger:
        """Configures and returns a structured JSON logger instance."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(StructuredJSONFormatter())
            logger.addHandler(handler)

        logger.propagate = False
        return logger


    logger = setup_logger()