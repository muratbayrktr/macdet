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
from backend.longformer.models import DetectionRequest
from backend.longformer.service import InferenceEngine as LongformerEngine
from backend.finetuned.service import InferenceEngine as FinetunedEngine
from backend.watermark.service import InferenceEngine as WatermarkEngine
from langdetect import detect
from contextlib import asynccontextmanager
from backend.models import EngineHub
from logging import getLogger
import os
import json
import httpx

# Set up logging
logger = getLogger(__name__)

# When the app starts, load the models
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Loading Longformer.")
    EngineHub["longformer"] = LongformerEngine()
    logger.info("Loading Bert.")
    EngineHub["finetuned"] = FinetunedEngine()
    logger.info("Loading Watermark.")
    EngineHub["watermark"] = WatermarkEngine()
    yield
    # Clean up the ML models and release the resources
    EngineHub.clear()

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
    lifespan=lifespan,
)

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Endpoint to render the root page with available testbeds for selection.
    Retrieves and structures testbeds data to be used in the template.
    """
    try:
        # Fetch testbeds from the data module
        testbeds_response = get_available_testbeds()

        # Validate the response from the testbeds retrieval
        if not testbeds_response or testbeds_response.get("status") != "success":
            raise HTTPException(status_code=500, detail="Failed to retrieve testbeds.")

        # Extract and structure testbeds data
        testbeds = testbeds_response.get("data", {}).get("testbeds", [])
        wilder_testbeds = testbeds_response.get("data", {}).get("wilder_testbeds", [])
        structured_testbeds = {
            "regular": testbeds,
            "wilder": wilder_testbeds,
        }

        # Debugging: Print structured testbeds (optional, for development purposes)
        # print("Structured Testbeds:", structured_testbeds)

        # Render the template with structured testbeds
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "title": "MACDET Home",
                "testbeds": structured_testbeds,
            },
        )
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions to propagate them as-is
        raise http_exc
    except Exception as exc:
        # Log unexpected errors (optional)
        # print(f"Unexpected error occurred: {exc}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(exc)}")

@app.post("/infer")
async def model_infer(request: Request):
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


@app.post("/inferAll")
async def general_infer(request: Request):
    """
    General inference endpoint that forwards the request to all models' inference endpoints.
    """
    try:
        # Parse the request body to extract the text
        payload = await request.json()
        text = payload.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required in the request payload.")
        
        # List of model names to query
        model_names = ["longformer", "finetuned", "watermark"]

        # Run all model inferences concurrently
        payloads = [{"model": model_name, "text": text} for model_name in model_names]
        responses = []
        async with httpx.AsyncClient() as client:
            for model_name, payload in zip(model_names, payloads):
                target_url = f"http://127.0.0.1:8000/{model_name}/infer"
                response = await client.post(target_url, json=payload)
                responses.append(response)
                print(response.status_code)
                if response.status_code == 200:
                    print(response.json())
                elif response.status_code == 307:
                    print("The model is not available.")
                else:
                    print(response.text)

        # Handle the responses from the models' endpoints
        # Example response should look like:
        # {
        #     "longformer": {
        #         "label": "machine-generated",
        #         "confidence": 0.89
        #     },
        #     "bert": {
        #         "label": "human-written",
        #         "confidence": 0.92
        #     },
        #     "watermark": {
        #         "label": "machine-generated",
        #         "confidence": 0.87,
        #         "text": "This is a watermarked sample text."
        #     }
        # }
        response = {
            model_name: response.json() if response.status_code == 200 else None
            for model_name, response in zip(model_names, responses)
        }
        ## Ensemble approach 
        # Step 1: detect language of the text
        # Step 2: if language is not english, return the response from finetuned model
        # Step 3: if language is english, return the response from ensemble of longformer and watermark models

        # Step 1
        # Detect language of the text
        response["macdet"] = {}
        language = detect(text)

        # Step 2
        # Ensemble of longformer and watermark models
        longformer_response = response["longformer"]
        watermark_response = response["watermark"]
        finetuned_response = response["finetuned"]
        # Extract values from responses
        longformer_confidence = longformer_response.get("confidence", 0.0)
        finetuned_confidence = finetuned_response.get("confidence", 0.0)
        watermark_confidence = watermark_response.get("confidence", 0.0)
        watermark_p_value = watermark_response.get("p_value", 1.0)
        watermark_z_score = watermark_response.get("z_score", 0.0)

        # Statistical significance based on multiple factors
        green_fraction_significance = watermark_response.get("green_fraction", 0) > 1.1 * 0.25  # Adjust based on γ
        is_watermark_significant = (watermark_p_value < 0.5) or (watermark_z_score > 1.5 and green_fraction_significance)

        # Dynamic weighting
        longformer_weight = longformer_confidence
        watermark_weight = (
            watermark_confidence * 1.5 if is_watermark_significant else watermark_confidence * 0.7
        )
        finetuned_weight = finetuned_confidence * 1.5 if language != "en" else finetuned_confidence * 0.4

        # Normalize weights
        total_weight = longformer_weight + watermark_weight + finetuned_weight
        if total_weight > 0:
            longformer_weight /= total_weight
            watermark_weight /= total_weight
            finetuned_weight /= total_weight
        else:
            longformer_weight = watermark_weight = finetuned_weight = 1.0 / 3.0

        # Combined confidence
        combined_confidence = (
            longformer_weight * longformer_confidence +
            watermark_weight * watermark_confidence +
            finetuned_weight * finetuned_confidence
        )

        # Decision logic
        if longformer_confidence > 0.996:
            final_prediction = longformer_response["label"]
        elif longformer_response["label"] == "machine-generated" and watermark_response["label"] == "machine-generated":
            final_prediction = "machine-generated"
        elif longformer_response["label"] == "machine-generated" or watermark_response["label"] == "machine-generated":
            if is_watermark_significant:
                final_prediction = watermark_response["label"]
            else:
                # Default to the model with higher confidence
                final_prediction = "machine-generated" if longformer_confidence > watermark_confidence else "human-written"
        else:
            final_prediction = "human-written"

        # Update response
        response["macdet"]["label"] = final_prediction
        response["macdet"]["confidence"] = combined_confidence
                
        print(json.dumps(response, indent=4))
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

        