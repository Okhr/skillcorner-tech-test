import structlog
import sys
import os


def setup_logger(log_path: str = None):
    processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        file = open(log_path, "a")
        logger_factory = structlog.WriteLoggerFactory(file)
    else:
        logger_factory = structlog.PrintLoggerFactory(sys.stdout)

    structlog.configure(
        processors=processors,
        logger_factory=logger_factory,
    )
    return structlog.get_logger()


# Default logger points to stdout
logger = setup_logger()
