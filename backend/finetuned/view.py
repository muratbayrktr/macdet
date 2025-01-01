from backend.finetuned import router
from backend.finetuned.models import DetectionRequest, DetectionResponse, Error
from backend.models import EngineHub

@router.post("/infer")
async def infer(request: DetectionRequest) -> DetectionResponse | Error:
    engine = EngineHub.get("finetuned", None)
    if engine is None:
        return Error(message="Engine not found.")
    return engine.predict(request)
