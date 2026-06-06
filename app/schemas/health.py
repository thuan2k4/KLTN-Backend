from pydantic import BaseModel
from typing import List

class HealthResponse(BaseModel):
    status: str
    mode: str
    device: str
    models_loaded: List[str]
    version: str
