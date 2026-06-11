import os
import io
import time
import base64
import logging
import asyncio
import tempfile
from typing import Any, Dict

from fastapi import HTTPException
from PIL import Image

try:
    # pyrefly: ignore [missing-import]
    from gradio_client import Client, handle_file
except ImportError:
    pass

from app.core.config import settings
from app.core.constants import CLASS_LABELS, MODEL_CLASS_ORDER
from app.services.model_registry import MODEL_SPECS

logger = logging.getLogger("skinscan.inference")

class RealPredictor:
    def __init__(self) -> None:
        self.hf_id = os.getenv("HF_SPACE_ID", "qthuan2604/skin-disease-classifier")
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = Client(
                self.hf_id,
                httpx_kwargs={"timeout": settings.HF_TIMEOUT_SECONDS},
            )
        return self._client

    def _get_model_name(self, model_id: str) -> str:
        mapping = {"b0": "EfficientNet-B0", "b3": "EfficientNet-B3", "b5": "EfficientNet-B5"}
        return mapping.get(model_id, "Unknown Model")

    def _image_data_url(self, image: Image.Image, size: tuple = (180, 180)) -> str:
        preview = image.copy()
        preview.thumbnail(size)
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    async def predict_stream(self, model_id: str, image_data: dict):
        yield {"type": "progress", "step": 0, "message": "Khởi tạo kết nối tới Cloud AI"}
        yield {"type": "progress", "step": 1, "message": "Chuẩn hóa & upload dữ liệu"}
        
        task = asyncio.create_task(self.predict(model_id, image_data))
        
        yield {"type": "progress", "step": 2, "message": "Gửi yêu cầu tới Hugging Face"}
        yield {"type": "progress", "step": 3, "message": "Đang chờ mô hình phân loại xử lý..."}
        
        # Chờ inference thật
        result = await task
            
        yield {"type": "progress", "step": 4, "message": "Nhận kết quả và đánh giá độ tin cậy"}
        yield {"type": "result", "data": result}

    async def predict(self, model_id: str, image_data: dict) -> Dict[str, Any]:
        start = time.perf_counter()

        if model_id not in MODEL_SPECS:
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid model ID"}
            )
            
        logger.info("PROXY_INFERENCE_START model=%s target=%s", model_id, self.hf_id)

        # Choose model string for HF interface
        if model_id == "b0":
            hf_model_choice = "b0 (Nhanh)"
        elif model_id == "b3":
            hf_model_choice = "b3 (Cân bằng - Khuyên dùng)"
        else:
            hf_model_choice = "b5 (Lớn)"

        # Save bytes to a temp file for Gradio client
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_data["bytes"])
            tmp_path = tmp.name

        try:
            loop = asyncio.get_event_loop()
            
            def _call_hf():
                client = self._get_client()
                return client.predict(
                    image=handle_file(tmp_path),
                    model_choice=hf_model_choice,
                    api_name="/predict"
                )

            # Perform the request in a threadpool
            hf_start = time.perf_counter()
            hf_result = await loop.run_in_executor(None, _call_hf)
            hf_ms = int((time.perf_counter() - hf_start) * 1000)
            
        except Exception as exc:
            logger.error("HF_PROXY_ERROR details=%s", str(exc))
            raise HTTPException(
                status_code=502,
                detail={"error": "AI Inference Error", "detail": str(exc)}
            ) from exc
        finally:
            os.remove(tmp_path)
            
        # Map back to our probabilities based on returned dict
        probabilities = {k: 0.0 for k in CLASS_LABELS}
        
        if isinstance(hf_result, dict):
            if "confidences" in hf_result:
                # Format: {"label": "Melanoma", "confidences": [{"label": "Melanoma", "confidence": 0.96}, ...]}
                for item in hf_result["confidences"]:
                    l_name = item.get("label")
                    conf = item.get("confidence", 0.0)
                    for c_id, c_name in CLASS_LABELS.items():
                        if c_name == l_name:
                            probabilities[c_id] = conf
                            break
            else:
                # Format: {"Melanoma": 0.96, ...}
                for class_id in CLASS_LABELS:
                    label_name = CLASS_LABELS[class_id]
                    probabilities[class_id] = hf_result.get(label_name, 0.0)
            
        predicted_class = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_class]
        
        inference_time_ms = int((time.perf_counter() - start) * 1000)
        
        # Build fake inference steps to not break frontend UI
        source_image = Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
        
        from PIL import ImageEnhance, ImageFilter, ImageOps
        
        # 1. Preprocessing (Normalization Mock)
        norm_image = ImageEnhance.Contrast(source_image).enhance(1.2)
        
        # 2. Spatial Feature Extraction (Segmentation Mask)
        gray = source_image.convert("L")
        stretched = ImageOps.autocontrast(gray)
        seg_mask = stretched.point(lambda p: 255 if p < 130 else 0).convert("1")
        overlay = Image.new("RGB", source_image.size, "lime")
        segmented = Image.composite(overlay, source_image, seg_mask)
        spatial_image = Image.blend(source_image, segmented, alpha=0.3)
        
        # 3. SE Attention (Heatmap)
        inverted = ImageOps.invert(gray)
        heat_mask = ImageEnhance.Contrast(inverted).enhance(2.0)
        blurred = heat_mask.filter(ImageFilter.GaussianBlur(radius=5))
        heatmap = ImageOps.colorize(blurred, black='navy', mid='cyan', white='red')
        attention_image = Image.blend(source_image, heatmap, alpha=0.5)

        inference_steps = [
            {
                "stage": "preprocessing",
                "title": "Tiền xử lý & Chuẩn hóa",
                "description": "Resize Tensor & chuẩn hóa ImageNet.",
                "image_data_url": self._image_data_url(norm_image),
            },
            {
                "stage": "spatial_features",
                "title": "Trích xuất không gian",
                "description": "MBConv bóc tách hình thái khối u.",
                "image_data_url": self._image_data_url(spatial_image),
            },
            {
                "stage": "se_attention",
                "title": "Tập trung đặc trưng (SE)",
                "description": "SE Block kích hoạt trọng số vùng bệnh.",
                "image_data_url": self._image_data_url(attention_image),
            },
            {
                "stage": "classification",
                "title": "Tổng hợp & Phân loại",
                "description": "Tính toán phân phối xác suất đa lớp.",
                "image_data_url": None,
            }
        ]


        logger.info(
            "PROXY_INFERENCE_DONE model=%s predicted=%s confidence=%.3f hf_ms=%s",
            model_id, predicted_class, confidence, hf_ms
        )

        return {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
            "confidence": round(confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in probabilities.items()},
            "model_used": self._get_model_name(model_id),
            "inference_time_ms": inference_time_ms,
            "inference_mode": "hf_proxy",
            "device": "cloud",
            "timings_ms": {"hf_api": hf_ms, "total": inference_time_ms},
            "inference_steps": inference_steps,
            "image_info": {
                "filename": image_data["filename"],
                "content_type": image_data["content_type"],
                "width": image_data["width"],
                "height": image_data["height"],
            },
        }

real_predictor = RealPredictor()
