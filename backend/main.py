from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import HTTPException
from fastapi.requests import Request
from backend.data.view import router as data_router
from backend.longformer.view import router as longformer_router
from backend.finetuned.view import router as finetuned_router
from backend.watermark.view import router as watermark_router
from backend.data.view import get_available_testbeds
from backend.longformer.models import DetectionRequest, DetectionResponse
import os
import httpx

app = FastAPI(
    title="MACDET API",
    description="API for MACDET",
    routes=[
        *longformer_router.routes,
        *data_router.routes,
        *finetuned_router.routes,
        *watermark_router.routes,
        
    ],
    dependencies=[],
)

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))



from fastapi import Request, HTTPException
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Testbeds are retrieved from the data module
    testbeds_response = get_available_testbeds()

    # Validate response status
    if testbeds_response.get("status") != "success":
        raise HTTPException(status_code=500, detail="Failed to retrieve testbeds.")

    # Extract and structure testbeds
    testbeds = testbeds_response.get("data", {}).get("testbeds", [])
    wilder_testbeds = testbeds_response.get("data", {}).get("wilder_testbeds", [])
    
    # Structure the testbeds data for the template
    structured_testbeds = {
        "regular": testbeds,
        "wilder": wilder_testbeds,
    }

    # Render the template
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "title": "MACDET Home",
            "testbeds": structured_testbeds,
        }
    )

@app.post("/infer")
async def general_infer(request: Request):
    """
    General inference endpoint that forwards the request to the appropriate model's inference endpoint.
    """
    try:
        # Parse the request body to extract model and data
        payload = await request.json()
        model_name = payload.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required in the request payload.")
        
        # Build the target URL for the model's inference endpoint
        target_url = f"http://127.0.0.1:8000/{model_name}/infer"

        print(target_url)
        print(payload)

        # Compile the detectionRequest
        detection_request = DetectionRequest(text=payload.get("text"))

        # Forward the request to the selected model's endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(target_url, json=detection_request.model_dump(mode="json"))

        print(response.status_code)
        # Handle the response from the model's endpoint
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 307:
            raise HTTPException(status_code=307, detail="The model is not available.")
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")