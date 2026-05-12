"""Shared graph state for the LangGraph agri-weather workflow."""
from typing import TypedDict

from typing_extensions import NotRequired

from src_langgraph.prompts import RiskReport


class AgriState(TypedDict):
    location_input: str
    forecast: NotRequired[str]              # filled by forecaster node
    analysis: NotRequired[RiskReport]       # filled by agronomist node
    final_report: NotRequired[RiskReport]   # filled by reviewer node
