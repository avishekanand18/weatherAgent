"""
LangGraph replica of the CrewAI agri-weather workflow.

Graph: START -> forecaster -> agronomist -> reviewer -> END

- forecaster: deterministic MCP tool call (no LLM).
- agronomist: Gemini LLM with structured RiskReport output.
- reviewer:   Gemini LLM auditing the draft, also structured output.
"""
from __future__ import annotations

# Make Python's ssl module use the OS (Windows) trust store. This is required
# behind corporate TLS interception, where the LangSmith httpx client would
# otherwise fail with "certificate verify failed: unable to get local issuer
# certificate". Must run before any module imports `ssl` and builds a context.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.tracers.context import collect_runs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from src_langgraph import prompts
from src_langgraph.prompts import RiskReport
from src_langgraph.state import AgriState# ---------------------------------------------------------------------------
# Paths + env + config bootstrap.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
MCP_SERVER_SCRIPT = str(_HERE / "mcp_server.py")
CONFIG_PATH = _HERE / "config.yaml"
LOGS_DIR = _PROJECT_ROOT / "logs"

load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

os.environ["GEMINI_API_KEY"] = google_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

LOGS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LangSmith integration helpers.
# ---------------------------------------------------------------------------
# A single thread_id ties every query in this REPL session together in the
# LangSmith "Threads" view, instead of producing N orphan runs.
_SESSION_ID = str(uuid.uuid4())
_LS_TRACING = os.getenv("LANGSMITH_TRACING", "").lower() == "true"

# Lightweight context tags pushed onto every run.
_RUN_METADATA_BASE = {
    "framework": "langgraph",
    "model": config["llm"]["model_name"],
    "temperature": config["llm"]["temperature"],
    "session_id": _SESSION_ID,
}

_llm = ChatGoogleGenerativeAI(
    model=config["llm"]["model_name"],
    temperature=config["llm"]["temperature"],
    google_api_key=google_api_key,
)


async def _build_graph():
    """Compile the LangGraph workflow with MCP tools wired in."""
    client = MultiServerMCPClient(
        {
            "weather": {
                "command": sys.executable,
                "args": [MCP_SERVER_SCRIPT],
                "transport": "stdio",
                "env": {**os.environ},
            }
        }
    )
    tools = await client.get_tools()
    forecast_tool = next(t for t in tools if t.name == "get_forecast_for_location")

    agronomist_llm = _llm.with_structured_output(RiskReport)
    reviewer_llm = _llm.with_structured_output(RiskReport)

    async def forecaster_node(state: AgriState) -> dict:
        loc = state["location_input"]
        result = await forecast_tool.ainvoke(
            {"location_name": loc},
            config={
                "run_name": f"MCP · forecast({loc})",
                "tags": ["mcp", "tool"],
            },
        )
        return {"forecast": str(result)}

    async def agronomist_node(state: AgriState) -> dict:
        forecast = state["forecast"]  # type: ignore[typeddict-item]
        msgs = [
            ("system", prompts.AGRONOMIST_BACKSTORY),
            (
                "user",
                prompts.get_analysis_task_desc()
                + f"\n\n14-DAY FORECAST:\n{forecast}",
            ),
        ]
        report = await agronomist_llm.ainvoke(
            msgs,
            config={
                "run_name": "Agronomist · analyse forecast",
                "tags": ["llm", "analysis"],
            },
        )
        return {"analysis": report}

    async def reviewer_node(state: AgriState) -> dict:
        forecast = state["forecast"]  # type: ignore[typeddict-item]
        draft: RiskReport = state["analysis"]  # type: ignore[typeddict-item]
        msgs = [
            ("system", prompts.REVIEWER_BACKSTORY),
            (
                "user",
                prompts.get_review_task_desc()
                + f"\n\nFORECAST:\n{forecast}"
                + f"\n\nDRAFT REPORT:\n{draft.model_dump_json(indent=2)}",
            ),
        ]
        report = await reviewer_llm.ainvoke(
            msgs,
            config={
                "run_name": "Reviewer · audit draft",
                "tags": ["llm", "reflection"],
            },
        )
        return {"final_report": report}

    g = StateGraph(AgriState)
    g.add_node("forecaster", forecaster_node)
    g.add_node("agronomist", agronomist_node)
    g.add_node("reviewer", reviewer_node)
    g.add_edge(START, "forecaster")
    g.add_edge("forecaster", "agronomist")
    g.add_edge("agronomist", "reviewer")
    g.add_edge("reviewer", END)
    return g.compile()


async def _run_async(graph, location_input: str) -> RiskReport:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = LOGS_DIR / f"session_langgraph_{timestamp}.txt"

    initial: AgriState = {"location_input": location_input}
    # Run name + tags + metadata + thread_id surface in the LangSmith UI when
    # LANGSMITH_TRACING=true. thread_id groups every query in this REPL
    # session under one "Thread" instead of producing orphan runs.
    run_config: dict = {
        "run_name": f"agri-weather: {location_input}",
        "tags": ["agri-weather", "langgraph"],
        "metadata": {**_RUN_METADATA_BASE, "location_input": location_input},
        "configurable": {"thread_id": _SESSION_ID},
    }

    # `collect_runs` captures the root run id so we can enrich it with outcome
    # metadata + feedback once the workflow has produced its final answer.
    with collect_runs() as cb:
        final_state = await graph.ainvoke(initial, config=run_config)

    report: RiskReport = final_state["final_report"]
    draft: RiskReport = final_state["analysis"]
    reflection_changed = report.model_dump() != draft.model_dump()

    log_file_path.write_text(
        "=== LangGraph session ===\n"
        f"Location: {location_input}\n\n"
        f"--- Forecast ---\n{final_state.get('forecast', '')}\n"
        f"--- Draft Analysis ---\n{draft.model_dump_json(indent=2)}\n"
        f"--- Final Report ---\n{report.model_dump_json(indent=2)}\n"
        f"--- Reflection changed report: {reflection_changed} ---\n",
        encoding="utf-8",
    )

    _annotate_run(cb, report, reflection_changed)
    return report


def _annotate_run(cb, report: RiskReport, reflection_changed: bool) -> None:
    """Push outcome metadata + a feedback score onto the root LangSmith run.

    Lets you filter runs in the LangSmith UI by `severity:*` / `risk:*` and
    chart the "reflection rate" (how often the Reviewer actually corrected
    the Agronomist's draft).
    """
    if not _LS_TRACING or not cb.traced_runs:
        return
    try:
        from langsmith import Client  # local import: optional dep
        client = Client()
        root_id = cb.traced_runs[0].id
        client.update_run(
            root_id,
            extra={
                "metadata": {
                    "primary_risk": report.primary_risk,
                    "severity": report.severity,
                    "affected_days": len(report.affected_dates),
                    "reflection_changed_report": reflection_changed,
                }
            },
            tags=[
                "agri-weather",
                "langgraph",
                f"risk:{report.primary_risk}",
                f"severity:{report.severity}",
            ],
        )
        client.create_feedback(
            root_id,
            key="reflection_triggered",
            score=1 if reflection_changed else 0,
            comment=(
                "Reviewer modified the agronomist's draft."
                if reflection_changed
                else "Reviewer approved the draft as-is."
            ),
        )
    except Exception as exc:
        # Never let observability break the user-facing run.
        print(f"[langsmith] failed to annotate root run: {exc}")


# ---------------------------------------------------------------------------
# Persistent event loop + cached compiled graph.
#
# Why: each call to asyncio.run() creates and destroys a fresh event loop.
# On Windows (ProactorEventLoop), the Gemini gRPC/HTTPS SSL connections
# opened by langchain-google-genai outlive that loop and try to flush during
# GC, producing "RuntimeError: Event loop is closed" / "Fatal error on SSL
# transport". Reusing one loop for the whole REPL session avoids the churn
# and lets connections shut down on the same loop that created them.
# ---------------------------------------------------------------------------
_LOOP: asyncio.AbstractEventLoop | None = None
_GRAPH = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
    return _LOOP


def run_agri_graph(location_input: str) -> RiskReport:
    """Sync entry point that mirrors `run_agri_crew`'s signature."""
    global _GRAPH
    loop = _get_loop()
    if _GRAPH is None:
        _GRAPH = loop.run_until_complete(_build_graph())
    return loop.run_until_complete(_run_async(_GRAPH, location_input))


def shutdown() -> None:
    """Cleanly close the persistent loop. Call once on program exit."""
    global _LOOP, _GRAPH
    if _LOOP is None or _LOOP.is_closed():
        return
    try:
        # Let pending callbacks (SSL shutdowns, transport closes) flush.
        _LOOP.run_until_complete(asyncio.sleep(0.25))
    except Exception:
        pass
    finally:
        _LOOP.close()
        _LOOP = None
        _GRAPH = None
