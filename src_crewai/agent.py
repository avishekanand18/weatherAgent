"""
Build and run the Agri-Weather multi-agent Crew.

Design notes
------------
- The geocoding + weather-fetch step is now a single deterministic MCP tool
  (`get_forecast_for_location`) consumed by ONE Forecaster agent. The previous
  two tool-wrapper agents added LLM cost without adding reasoning value.
- The Agronomist returns a Pydantic `RiskReport`, which is then audited by a
  Reviewer agent (reflection / self-critique step).
- Memory is OFF; tasks pass context explicitly via the `context=` parameter.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv
from mcp import StdioServerParameters

from src_crewai import prompts
from src_crewai.prompts import RiskReport

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file - works from any cwd).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
MCP_SERVER_SCRIPT = str(_HERE / "mcp_server.py")
CONFIG_PATH = _HERE / "config.yaml"
LOGS_DIR = _PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Env + config bootstrap.
# ---------------------------------------------------------------------------
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

os.environ["GEMINI_API_KEY"] = google_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

llm = LLM(
    model=f"gemini/{config['llm']['model_name']}",
    temperature=config['llm']['temperature'],
    api_key=google_api_key,
)

LOGS_DIR.mkdir(exist_ok=True)


def _pick_tool(mcp_tools, name: str):
    """Look up a tool exposed by the MCP server by its registered name."""
    try:
        return mcp_tools[name]
    except (TypeError, KeyError):
        for t in mcp_tools:
            if getattr(t, "name", None) == name:
                return t
    raise ValueError(
        f"Tool '{name}' not found on MCP server. Available: "
        f"{[getattr(t, 'name', '?') for t in mcp_tools]}"
    )


def run_agri_crew(location_input: str):
    """
    Build and run the multi-agent Crew for the given location.

    Returns the final `RiskReport` (Pydantic model) from the Reviewer task.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = str(LOGS_DIR / f"session_{timestamp}.txt")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
        env={**os.environ},
    )

    with MCPServerAdapter(server_params) as mcp_tools:
        forecast_tool = _pick_tool(mcp_tools, "get_forecast_for_location")

        # ---------------- AGENTS ----------------
        forecaster = Agent(
            role="Agri-Weather Forecaster",
            goal="Retrieve the raw 14-day agricultural forecast for the requested location.",
            backstory=(
                "You are a thin wrapper over the get_forecast_for_location MCP tool. "
                "Always call the tool exactly once with the user-supplied location string "
                "and return its output verbatim. Do not interpret the data."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[forecast_tool],
            llm=llm,
        )

        agronomist = Agent(
            role="Senior Agronomist",
            goal="Analyse the 14-day forecast for vineyard risks and emit a structured RiskReport.",
            backstory=prompts.AGRONOMIST_BACKSTORY,
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        reviewer = Agent(
            role="Chief Agronomy Reviewer",
            goal="Audit the agronomist's RiskReport against the underlying forecast and correct it if needed.",
            backstory=prompts.REVIEWER_BACKSTORY,
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        # ---------------- TASKS ----------------
        task_forecast = Task(
            description=prompts.get_forecast_task_desc(location_input),
            expected_output=prompts.FORECAST_TASK_EXPECTED,
            agent=forecaster,
        )

        task_analysis = Task(
            description=prompts.get_analysis_task_desc(),
            expected_output=prompts.ANALYSIS_TASK_EXPECTED,
            agent=agronomist,
            context=[task_forecast],          # explicit context, no Crew memory
            output_pydantic=RiskReport,       # structured output
        )

        task_review = Task(
            description=prompts.get_review_task_desc(),
            expected_output=prompts.REVIEW_TASK_EXPECTED,
            agent=reviewer,
            context=[task_forecast, task_analysis],
            output_pydantic=RiskReport,
        )

        # ---------------- CREW ----------------
        crew = Crew(
            agents=[forecaster, agronomist, reviewer],
            tasks=[task_forecast, task_analysis, task_review],
            process=Process.sequential,
            verbose=True,
            memory=False,                     # context passed explicitly
            output_log_file=log_file_path,
        )

        return crew.kickoff(inputs={"location_input": location_input})
