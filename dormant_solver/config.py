"""Configuration and logging for Dormant LLM puzzle solver."""

import logging
from pathlib import Path

API_KEY = "a8431788-a08f-457c-b3a4-660d06576579"
MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
WARMUP_MODEL = "dormant-model-warmup"
ALL_MODELS = [WARMUP_MODEL] + MODELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# DeepSeek V3 architecture constants
DSV3_LAYERS = 61  # layers 0-60
DSV3_DENSE_LAYERS = range(0, 3)  # layers 0-2 dense MLP
DSV3_MOE_LAYERS = range(3, 61)  # layers 3-60 MoE
DSV3_NUM_EXPERTS = 256
DSV3_ACTIVE_EXPERTS = 8

# Qwen2 warmup constants
QWEN2_LAYERS = 28

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "solver.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
