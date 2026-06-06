from pathlib import Path
from typing import Dict, List

from app.core.config import settings
from app.core.constants import MODELS_METADATA


MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "b0": {
        "architecture": "efficientnet_b0",
        "input_size": 300,
    },
    "b3": {
        "architecture": "efficientnet_b3",
        "input_size": 300,
    },
    "b5": {
        "architecture": "efficientnet_b5",
        "input_size": 300,
    },
}


def available_model_ids() -> List[str]:
    # Since models are hosted on Hugging Face, they are all available
    return [model["id"] for model in MODELS_METADATA]


def models_metadata_with_availability() -> List[dict]:
    return [
        {
            **model,
            "available": True,
        }
        for model in MODELS_METADATA
    ]
