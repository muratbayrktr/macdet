""" This module contains the type hints for the assistant module. """

from __future__ import annotations

from enum import Enum


class NewsSource(str, Enum):
    """An enumeration of news sources."""

    BBC_NEWS = "bbc-news"
    BLOOMBERG = "bloomberg"


class NewsCategory(str, Enum):
    """An enumeration of financial news categories."""

    METALS = "precious-metals"
    CRYPTO = "crypto"
    STOCKS = "stocks"
    BONDS = "bonds"
    COMMODITIES = "commodities"
    CURRENCIES = "currencies"
    MACRO = "macro"
