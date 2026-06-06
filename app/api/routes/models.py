from fastapi import APIRouter
from app.schemas.model import ModelListResponse
from app.core.config import settings
from app.services.model_registry import models_metadata_with_availability

router = APIRouter()

@router.get("/models", response_model=ModelListResponse)
async def get_models():
    return {
        "models": models_metadata_with_availability(),
        "default": settings.DEFAULT_MODEL
    }
