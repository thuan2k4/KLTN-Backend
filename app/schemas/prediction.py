from pydantic import BaseModel
from typing import Dict, List, Optional

class ImageInfo(BaseModel):
    filename: str
    content_type: str
    width: int
    height: int

class InferenceStep(BaseModel):
    stage: str
    title: str
    description: str
    image_data_url: Optional[str] = None

class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    inference_time_ms: int
    image_info: ImageInfo
    inference_mode: str
    device: str
    timings_ms: Optional[Dict[str, int]] = None
    inference_steps: Optional[List[InferenceStep]] = None
