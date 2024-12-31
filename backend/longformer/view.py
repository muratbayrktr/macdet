
from backend.longformer import router
from backend.longformer.models import DetectionRequest, DetectionResponse, Error
from backend.models import EngineHub

@router.post("/infer")
async def infer(request: DetectionRequest) -> DetectionResponse | Error:
    engine = EngineHub.get("longformer", None)
    if engine is None:
        return Error(message="Engine not found.")
    return engine.predict(request)
