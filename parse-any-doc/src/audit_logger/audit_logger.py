import os
from loguru import logger
import sys
from copy import copy

"""
Setup logger configuration.
"""
logger.remove()
log_name = os.getenv("LOG_FILE", os.getcwd() + "/audit.log")
print(f"** log_name -> {log_name}")
if log_name:
    logger.add(
        log_name,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        retention="2 days",
        rotation="5 MB", 
    )
else:
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

logger_audit_handler = logger #copy(logger)
