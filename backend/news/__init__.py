""" This module is used to define the news routes. """

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/news", tags=["news"])
