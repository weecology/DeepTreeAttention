"""Lightning logger stub for local / CI runs without Comet."""

from __future__ import annotations

from lightning_fabric.utilities import rank_zero_only
from pytorch_lightning.loggers import Logger


class _SmokeExperiment:
    id = "smoke"

    def get_key(self) -> str:
        return "smoke"

    def log_parameter(self, *args, **kwargs):
        pass

    def add_tag(self, *args, **kwargs):
        pass

    def log_parameters(self, *args, **kwargs):
        pass

    def log_table(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_metric(self, *args, **kwargs):
        pass

    def log_confusion_matrix(self, *args, **kwargs):
        pass


class LocalSmokeLogger(Logger):
    """Minimal logger so ``train.py`` can call ``.experiment.*`` without Comet."""

    def __init__(self) -> None:
        super().__init__()
        self._experiment = _SmokeExperiment()

    @property
    def experiment(self) -> _SmokeExperiment:
        return self._experiment

    @property
    def name(self) -> str:
        return "local_smoke"

    @property
    def version(self) -> str:
        return "0"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass
