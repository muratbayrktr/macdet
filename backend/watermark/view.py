from backend.watermark import router
from backend.watermark.models import DetectionRequest, DetectionResponse, Error
from backend.watermark.service import InferenceEngine

@router.post("/infer")
async def infer(request: DetectionRequest) -> DetectionResponse | Error:
    engine = InferenceEngine()
    return engine.predict(request)
