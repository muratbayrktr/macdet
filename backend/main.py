from __future__ import annotations

from fastapi.responses import HTMLResponse

from fastapi import FastAPI
from backend.longformer.view import router as longformer_router

app = FastAPI(
    title="MACDET API",
    description="API for MACDET",
    routes=[
        *longformer_router.routes,
    ],
    dependencies=[],
)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content="<h1>Longformer</h1>", status_code=200)