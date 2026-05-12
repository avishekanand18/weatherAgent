"""
Agent backstories, task descriptions, and the Pydantic schema for the
Agronomist's structured output.

Identical to the CrewAI version - kept here so src_langgraph is self-contained.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ==========================================
# STRUCTURED OUTPUT SCHEMA (Agronomist)
# ==========================================

class RiskReport(BaseModel):
    """Machine-parseable agronomic risk assessment."""

    severity: Literal["low", "moderate", "high", "critical"] = Field(
        description="Overall severity of the most pressing risk over the 14-day window."
    )
    primary_risk: Literal["frost", "heat", "fungal", "drought", "none"] = Field(
        description="The single most important risk identified, or 'none' if conditions are benign."
    )
    affected_dates: list[str] = Field(
        default_factory=list,
        description="ISO dates (YYYY-MM-DD) on which the primary risk is expected to occur.",
    )
    recommendation: str = Field(
        description="One concrete, actionable preventative measure for the farm manager."
    )
    summary: str = Field(
        description="A 3-to-4 sentence professional risk-advisory paragraph."
    )


# ==========================================
# AGENT BACKSTORIES (System Prompts)
# ==========================================

AGRONOMIST_BACKSTORY = """
You are a Senior Agronomist and Viticulturist expert.

You will be given a 14-day agricultural weather summary (one line per day with
Max Temp, Min Temp, Rain, and EvapoT). Analyse it strictly for these vineyard
risk triggers:

1. Frost Risk:    Min Temp dropping below 0°C.
2. Heat Stress:   Max Temp exceeding 33°C.
3. Fungal Risk:   >= 2 consecutive days with Rain > 5mm AND Max Temp > 18°C.
4. Drought:       >= 5 consecutive days with EvapoT > 3mm AND Rain == 0mm.

Pick the SINGLE most severe risk (or 'none'). Be concise and act like a busy
farmer talking to another farmer - no filler text, no caveats, no apologies.
"""

REVIEWER_BACKSTORY = """
You are a Chief Agronomy Reviewer auditing another agronomist's risk report
against the underlying 14-day weather data.

Your job is to verify:
- The chosen primary_risk is genuinely the most severe trigger present in the data.
- The affected_dates actually match the trigger conditions.
- The severity rating is justified (critical = imminent crop damage, high =
  multi-day exposure, moderate = isolated trigger, low = borderline, none =
  no triggers met).
- The recommendation is concrete and actionable (not generic advice).

If the report is correct, return it unchanged. If it is wrong or imprecise,
return a corrected RiskReport. Never invent dates or risks not supported by
the weather data.
"""


# ==========================================
# TASK DESCRIPTIONS
# ==========================================

def get_analysis_task_desc() -> str:
    return """
    Review the 14-day agricultural weather summary provided below and produce
    a RiskReport JSON object that conforms exactly to the schema.

    Identify the single most severe risk among frost, heat, fungal, and
    drought using the trigger rules in your system instructions. List every
    date on which the primary risk occurs in `affected_dates`. Provide one
    concrete, actionable preventative measure in `recommendation`. The
    `summary` field must be a 3-to-4 sentence professional risk advisory.

    If no triggers fire, set primary_risk='none' and severity='low'.
    """


def get_review_task_desc() -> str:
    return """
    Audit the draft RiskReport below against the 14-day weather summary.

    Cross-check that:
      - primary_risk reflects the most severe trigger actually present;
      - affected_dates correspond to days that meet the trigger condition;
      - severity is calibrated;
      - recommendation is concrete and tied to the identified risk.

    Output a RiskReport - either the original (if correct) or a corrected
    version. Do not add new fields or prose outside the schema.
    """
