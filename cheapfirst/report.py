"""cheapfirst — report metriche (alias per metrics.report)."""

from .metrics import MetricsLogger


def generate_report(db_path: str, days: int = 7) -> str:
    """Genera report in formato testo."""
    logger = MetricsLogger(db_path)
    return logger.report(days)
