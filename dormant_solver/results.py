"""Result storage structures."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import RESULTS_DIR, log


@dataclass
class ExperimentResult:
    experiment_id: str
    model: str
    description: str
    inputs: list
    outputs: list
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def save(self, path: Optional[Path] = None):
        path = path or RESULTS_DIR / f"{self.experiment_id}_{self.model}.json"
        data = {
            "experiment_id": self.experiment_id,
            "model": self.model,
            "description": self.description,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "results": [
                {"input": inp, "output": out}
                for inp, out in zip(self.inputs, self.outputs)
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"Saved {self.experiment_id} -> {path}")
