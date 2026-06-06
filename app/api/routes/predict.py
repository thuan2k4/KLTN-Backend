import json
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.prediction import PredictionResponse
from app.schemas.error import ErrorResponse
from app.services.image_validator import validate_image
from app.services.predictor import get_prediction
from app.core.constants import MODELS_METADATA

router = APIRouter()

@router.post("/predict", 
             response_model=PredictionResponse,
             responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict(
    file: UploadFile = File(...),
    model: str = Form("b3"),
    mock_scenario: str = Query(None)
):
    # 1. Validate model id
    valid_models = [m["id"] for m in MODELS_METADATA]
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid model ID",
                "detail": f"Supported models: {', '.join(valid_models)}",
                "code": "INVALID_MODEL_ID"
            }
        )

    # 2. Validate image
    image_data = await validate_image(file)

    # 3. Get prediction
    try:
        result = await get_prediction(model, image_data, mock_scenario)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Inference error",
                "detail": str(e),
                "code": "INFERENCE_ERROR"
            }
        )

@router.post("/predict/stream")
async def predict_stream_endpoint(
    file: UploadFile = File(...),
    model: str = Form("b3"),
    mock_scenario: str = Query(None)
):
    valid_models = [m["id"] for m in MODELS_METADATA]
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid model ID", "detail": f"Supported models: {', '.join(valid_models)}", "code": "INVALID_MODEL_ID"}
        )

    image_data = await validate_image(file)
    from app.services.predictor import get_prediction_stream

    async def event_generator():
        try:
            async for event in get_prediction_stream(model, image_data, mock_scenario):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
