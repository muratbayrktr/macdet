from pydantic import BaseModel

class DetectionRequest(BaseModel):
    text: str

# {
#     "label": "machine-generated",
#     "confidence": 0.9999,
#     "num_tokens_scored": 264,
#     "num_green_tokens": 93,
#     "green_fraction": 0.3523,
#     "z_score": 3.8376,
#     "p_value": 0.00006212,
#     "detection_threshold": 4.0
# }
class WatermarkDetectionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    num_tokens_scored: int | None = None
    num_green_tokens: int | None = None
    green_fraction: float | None = None
    z_score: float | None = None
    p_value: float | None = None
    detection_threshold: float | None = None


class Error(BaseModel):
    message: str
