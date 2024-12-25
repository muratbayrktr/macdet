from backend.finetuned import router
from backend.finetuned.models import DetectionRequest, DetectionResponse, Error
from backend.longformer.service import InferenceEngine

@router.post("/infer")
async def infer(request: DetectionRequest) -> DetectionResponse | Error:
    engine = InferenceEngine()
    return engine.predict(request)
