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
import ranx
from ranx import Qrels, Run, fuse
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
                # favicon
                "favicon": "/static/favicon.ico",
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
            model_name: r.json() if r.status_code == 200 else None
            for model_name, r in zip(model_names, responses)
        }
        
        base_weights = {
            "longformer": 1.2,
            "finetuned": 1.0,
            "watermark": 0.8
        }

        # Label <-> index mapping for a 2-class problem
        label_to_index = {"machine-generated": 0, "human-written": 1}
        index_to_label = {0: "machine-generated", 1: "human-written"}

        # Initialize combined distribution
        fused_scores = [0.0, 0.0]  # [machine, human]

        # A helper to ensure numeric stability in case we need to softmax
        # (Optional if your logprobs already sum to 1.0, but included for safety.)
        def _safe_normalize(prob_list):
            total = sum(prob_list)
            return [p / total for p in prob_list] if total > 0 else prob_list

        # A helper to extract [p_machine, p_human] from each model response
        def get_distribution(model_name: str, data: dict) -> list[float]:
            # If the model provides logprobs, we assume they're [p_machine, p_human]
            # or near-probabilities that just need normalization.
            if "logprobs" in data and isinstance(data["logprobs"], list):
                probs = _safe_normalize(data["logprobs"])
                return probs
            
            # Otherwise, fall back to label+confidence
            # We'll interpret "confidence" as belonging to the reported label.
            # For example, if label="machine-generated" and confidence=0.9,
            # then p_machine=0.9, p_human=0.1.
            if "label" in data and "confidence" in data:
                label = data["label"]
                conf = data["confidence"]
                if label == "machine-generated":
                    return [conf, 1 - conf]  # [p_machine, p_human]
                else:
                    return [1 - conf, conf]  # [p_machine, p_human]

            # If nothing is valid, return a uniform guess
            return [0.5, 0.5]

        # For each model, pull out a probability distribution and multiply by its weight
        for model_name in ["longformer", "finetuned", "watermark"]:
            model_data = response.get(model_name, {})
            w = base_weights.get(model_name, 1.0)
            dist = get_distribution(model_name, model_data)
            # Scale by the model's weight, then add to fused_scores
            fused_scores[0] += dist[0] * w
            fused_scores[1] += dist[1] * w

        # Normalize the fused distribution
        fused_scores = _safe_normalize(fused_scores)

        # Pick the final label and confidence
        final_index = 0 if fused_scores[0] > fused_scores[1] else 1
        final_label = index_to_label[final_index]
        final_confidence = fused_scores[final_index]

        response["macdet"] = {
            "label": final_label,
            "confidence": final_confidence,
            "logprobs": fused_scores
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

        