import random
import time
from typing import Dict, Any
from app.core.constants import CLASS_LABELS

class MockPredictor:
    def __init__(self):
        self.classes = list(CLASS_LABELS.keys())

    async def predict_stream(self, model_id: str, scenario: str = None):
        yield {"type": "progress", "step": 0, "message": "Khởi tạo pipeline phân tích"}
        yield {"type": "progress", "step": 1, "message": "Chuẩn hóa & xử lý ảnh (Tensor)"}
        yield {"type": "progress", "step": 2, "message": "Trích xuất đặc trưng (Feature Extraction)"}
        yield {"type": "progress", "step": 3, "message": "Phân loại bằng mô hình Deep Learning"}
        
        result = await self.predict(model_id, scenario)
        
        yield {"type": "progress", "step": 4, "message": "Hoàn tất & đánh giá độ tin cậy"}
        yield {"type": "result", "data": result}

    async def predict(self, model_id: str, scenario: str = None) -> Dict[str, Any]:
        # Giả lập thời gian inference
        inference_time = random.randint(80, 250)
        time.sleep(inference_time / 1000.0)

        # Logic tạo xác suất giả lập
        probs = self._generate_probabilities(scenario)
        
        # Lấy class cao nhất
        predicted_class = max(probs, key=probs.get)
        predicted_label = CLASS_LABELS[predicted_class]
        confidence = probs[predicted_class]

        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in probs.items()},
            "model_used": self._get_model_name(model_id),
            "inference_time_ms": inference_time,
            "inference_mode": "mock",
            "device": "mock",
        }

    def _generate_probabilities(self, scenario: str) -> Dict[str, float]:
        probs = {c: random.uniform(0.01, 0.05) for c in self.classes}
        
        target_class = None
        if scenario == "cancer_high_confidence":
            target_class = random.choice(["MEL", "BCC", "SCC"])
            probs[target_class] = random.uniform(0.8, 0.95)
        elif scenario == "benign_high_confidence":
            target_class = random.choice(["NV", "BKL", "DF", "VASC"])
            probs[target_class] = random.uniform(0.8, 0.95)
        elif scenario == "precancer_medium_confidence":
            target_class = "AK"
            probs[target_class] = random.uniform(0.5, 0.7)
        else:
            # Random scenario
            target_class = random.choice(self.classes)
            probs[target_class] = random.uniform(0.4, 0.6)

        # Normalize to sum up to ~1.0
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def _get_model_name(self, model_id: str) -> str:
        mapping = {"b0": "EfficientNet-B0", "b3": "EfficientNet-B3", "b5": "EfficientNet-B5"}
        return mapping.get(model_id, "Unknown Model")

mock_predictor = MockPredictor()
