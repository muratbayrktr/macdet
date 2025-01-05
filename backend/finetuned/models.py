from pydantic import BaseModel

class DetectionRequest(BaseModel):
    text: str

class DetectionResponse(BaseModel):
    label: str
    confidence: float
    logprobs: list

class Error(BaseModel):
    message: str
