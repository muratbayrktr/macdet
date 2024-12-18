""" This module contains the service class for the news app. """

from __future__ import annotations

# from app.news.schemas import Article, NewsRequest, NewsResponse


class NewsService:
    """A class to interact with the news service."""

    def __init__(self):
        pass

    # def pull_news(self, request: NewsRequest) -> NewsResponse:
    #     """Pull news from the internet based on a query."""
    #     url = "https://api.marketaux.com//v1/news/all?countries=us&\
    #         filter_entities=true&limit=10&published_after=2024-10-23T11:55&\
    #             api_token=UekNdF56YYAfG1wAQTZzFMuMKWLAhLwgVQcvKvCc"

    #     response = requests.get(url=url)
    #     print(response.json())
    #     if response.status_code == 200:
    #         return NewsResponse(
    #             articles=list(response.json()['data'])
    #         )

    #     return NewsResponse(
    #         articles=[
    #             Article(
    #                 title="Test Title",
    #                 description="Test Description",
    #                 url="https://www.test.com",
    #                 published_at="2021-01-01",
    #                 source=request.source,
    #                 category=request.category,
    #                 sentiment="Positive",
    #                 sentiment_score=0.9,
    #                 reasoning="Test Reasoning",
    #             ),
    #             Article(
    #                 title="Test Title",
    #                 description="Test Description",
    #                 url="https://www.test.com",
    #                 published_at="2021-01-01",
    #                 source=request.source,
    #                 category=request.category,
    #                 sentiment="Positive",
    #                 sentiment_score=0.9,
    #                 reasoning="Test Reasoning",
    #             ),
    #         ],
    #     )

    def get_last_updated(self):
        """Get the last updated date for the news."""
        return "2021-01-01"
