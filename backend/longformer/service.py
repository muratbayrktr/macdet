from backend.longformer.utils import preprocess, detect
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from backend.longformer.models import DetectionRequest, DetectionResponse, Error
# conditionally import torch if available in the environment
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False



class InferenceEngine:
    def __init__(self):
        self.device = "cuda" if torch_available and torch.cuda.is_available() else "cpu"
        self.model_dir = "yaful/MAGE"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)

    def predict(self, detection_request: DetectionRequest) -> DetectionResponse:
        if not torch_available:
            raise ImportError("torch is not available in the environment.")
        inputs = preprocess(detection_request.text)
        outputs = detect(inputs, self.tokenizer, self.model, self.device)
        return DetectionResponse(
            **outputs
        )