from pydantic import BaseModel
from typing import List, Optional

class ModelMetadata(BaseModel):
    id: str
    name: str
    params: str
    input_size: int
    available: bool
    recommended: bool
    description: str

class ModelListResponse(BaseModel):
    models: List[ModelMetadata]
    default: str
