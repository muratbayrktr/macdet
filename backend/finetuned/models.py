from pydantic import BaseModel

class DetectionRequest(BaseModel):
    text: str

class DetectionResponse(BaseModel):
    text: str
    label: str
    confidence: float

class Error(BaseModel):
    message: str
