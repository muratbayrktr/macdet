from backend.watermark import router
from backend.watermark.models import DetectionRequest, WatermarkDetectionResponse, Error
from backend.models import EngineHub

@router.post("/infer")
async def infer(request: DetectionRequest) -> WatermarkDetectionResponse | Error:
    engine = EngineHub.get("watermark", None)
    if engine is None:
        return Error(message="Engine not found.")
    return engine.predict(request)
