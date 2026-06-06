from fastapi import APIRouter
from app.schemas.health import HealthResponse
from app.core.config import settings
from app.services.model_registry import available_model_ids

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "mode": "mock" if settings.USE_MOCK_INFERENCE else "real",
        "device": settings.MODEL_DEVICE,
        "models_loaded": available_model_ids(),
        "version": settings.APP_VERSION
    }
