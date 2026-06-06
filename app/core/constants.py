# ISIC 2019 Class Labels
CLASS_LABELS = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma"
}

# Notebook training order. Keep this separate from display label order because
# model logits must be mapped back using the exact class index order.
MODEL_CLASS_ORDER = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]

# Disease Mapping Types
CLASS_MAPPING = {
    "cancer": ["MEL", "BCC", "SCC"],
    "precancer": ["AK"],
    "benign": ["NV", "BKL", "DF", "VASC"]
}

# Model Metadata
MODELS_METADATA = [
    {
        "id": "b0",
        "name": "EfficientNet-B0",
        "params": "5.3M",
        "input_size": 300,
        "available": True,
        "recommended": False,
        "description": "Fastest model for quick screening"
    },
    {
        "id": "b3",
        "name": "EfficientNet-B3",
        "params": "12M",
        "input_size": 300,
        "available": True,
        "recommended": True,
        "description": "Balanced model for demo inference"
    },
    {
        "id": "b5",
        "name": "EfficientNet-B5",
        "params": "30M",
        "input_size": 300,
        "available": True,
        "recommended": False,
        "description": "Larger model with higher capacity"
    }
]

# Mock Scenarios
MOCK_SCENARIOS = [
    "cancer_high_confidence",
    "benign_high_confidence",
    "precancer_medium_confidence",
    "low_confidence"
]
