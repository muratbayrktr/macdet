"""
Finetuned Models Module.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/finetuned", tags=["finetuned"])
