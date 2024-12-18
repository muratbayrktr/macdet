""" This module defines the API routes for the FastAPI application. """

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query
from sqlmodel import select

from app.auth.utils import get_current_user
from app.database.conn import SessionDep
from app.news import router
from app.news.models import NewsArticle
from app.user.models import User


@router.post("/news/")
async def create_news(
    news: NewsArticle,
    session: SessionDep,
    current_user: User = Depends(get_current_user),
) -> NewsArticle:
    session.add(news)
    await session.commit()
    await session.refresh(news)
    return news


@router.get("/news/")
async def read_news(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
    current_user: User = Depends(get_current_user),
) -> list[NewsArticle]:
    news = (await session.exec(select(NewsArticle).offset(offset).limit(limit))).all()
    return news


@router.get("/news/{news_id}")
async def read_news_by_id(
    news_id: str,
    session: SessionDep,
    current_user: User = Depends(get_current_user),
) -> NewsArticle:
    news = await session.get(NewsArticle, news_id)
    return news


@router.delete("/news/{news_id}")
async def delete_news(
    news_id: str,
    session: SessionDep,
    current_user: User = Depends(get_current_user),
) -> NewsArticle:
    news = await session.get(NewsArticle, news_id)
    await session.delete(news)
    await session.commit()
    return news
