from __future__ import annotations

import uuid
from datetime import datetime

from sqlmodel import Field, SQLModel

from app.database.utils import get_naive_utc_now

# from app.database.utils import pydantic_to_sqlalchemy_model
# from app.news.schemas import NewsArticleSchema

# NewsArticle = pydantic_to_sqlalchemy_model(
#     NewsArticleSchema.NewsArticleBase,
#     "news_articles",
# )


class NewsArticle(SQLModel, table=True):
    __tablename__ = "news_articles"
    id: uuid.UUID = Field(default=uuid.uuid4, primary_key=True, index=True)
    title: str | None = Field(default=None, index=True)
    description: str | None = Field(default=None)
    url: str | None = Field(default=None)
    source: str | None = Field(default=None)
    category: str | None = Field(default=None, index=True)
    content: str | None = Field(default=None, index=True)
    published_at: datetime = Field(default_factory=get_naive_utc_now)
