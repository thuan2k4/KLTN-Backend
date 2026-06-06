import logging

from app.core.config import settings
from app.services.mock_predictor import mock_predictor

logger = logging.getLogger("skinscan.inference")


async def get_prediction(model_id: str, image_data: dict, scenario: str = None):
    if settings.USE_MOCK_INFERENCE:
        logger.warning(
            "MOCK_INFERENCE model=%s filename=%s",
            model_id,
            image_data.get("filename"),
        )
        result = await mock_predictor.predict(model_id, scenario)
        result["image_info"] = {
            "filename": image_data["filename"],
            "content_type": image_data["content_type"],
            "width": image_data["width"],
            "height": image_data["height"],
        }
        result["inference_mode"] = "mock"
        result["device"] = "mock"
        return result

    from app.services.real_predictor import real_predictor

    logger.info(
        "REAL_INFERENCE_REQUEST model=%s filename=%s bytes=%s",
        model_id,
        image_data.get("filename"),
        len(image_data.get("bytes", b"")),
    )
    return await real_predictor.predict(model_id, image_data)

async def get_prediction_stream(model_id: str, image_data: dict, scenario: str = None):
    if settings.USE_MOCK_INFERENCE:
        logger.warning("MOCK_INFERENCE_STREAM model=%s", model_id)
        async for item in mock_predictor.predict_stream(model_id, scenario):
            if item["type"] == "result":
                item["data"]["image_info"] = {
                    "filename": image_data["filename"],
                    "content_type": image_data["content_type"],
                    "width": image_data["width"],
                    "height": image_data["height"],
                }
            yield item
    else:
        from app.services.real_predictor import real_predictor
        logger.info("REAL_INFERENCE_STREAM_REQUEST model=%s", model_id)
        async for item in real_predictor.predict_stream(model_id, image_data):
            yield item
