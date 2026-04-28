import os
import sys
from datetime import datetime
from pathlib import Path
import yaml
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
# from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

from src import prompts

# Absolute path to the FastMCP server script (launched as a subprocess).
MCP_SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_server.py")

# 1. Load Environment Variables (.env)
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

os.environ["GEMINI_API_KEY"] = google_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key
# os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

if not google_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

# 2. Load Configuration (config.yaml)
with open("src/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# 3. Initialize the LLM using config parameters
llm = LLM(
    model=f"gemini/{config['llm']['model_name']}",
    temperature=config['llm']['temperature'],
    api_key=google_api_key
)

# 4. Setup Logging Directory and Timestamped File
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"logs/session_{timestamp}.txt"

def _pick_tool(mcp_tools, name: str):
    """Look up a tool exposed by the MCP server by its registered name."""
    # MCPServerAdapter exposes the tool list; support both dict-like and
    # iterable access patterns across crewai-tools versions.
    try:
        return mcp_tools[name]
    except (TypeError, KeyError):
        for t in mcp_tools:
            if getattr(t, "name", None) == name:
                return t
    raise ValueError(f"Tool '{name}' not found on MCP server. Available: "
                     f"{[getattr(t, 'name', '?') for t in mcp_tools]}")


def run_agri_crew(location_input: str):
    """
    Build and run the multi-agent Crew for the given location.

    Tools are served by a FastMCP server launched as a subprocess over stdio.
    The MCP adapter context must remain open for the lifetime of `kickoff`,
    so crew construction and execution both happen inside the `with` block.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
        env={**os.environ},
    )

    with MCPServerAdapter(server_params) as mcp_tools:
        geocoding_tool = _pick_tool(mcp_tools, "get_coordinates")
        weather_tool = _pick_tool(mcp_tools, "get_agri_weather")

        # ==========================================
        # AGENTS
        # ==========================================

        geocoder = Agent(
            role='Geolocation Specialist',
            goal='Find exact latitude and longitude for the location.',
            backstory=prompts.GEOCODER_BACKSTORY,
            verbose=True, # Requirement 2: Show traces in console
            allow_delegation=False,
            tools=[geocoding_tool],
            llm=llm
        )

        meteorologist = Agent(
            role='Agricultural Meteorologist',
            goal='Fetch hyper-local 14-day weather data.',
            backstory=prompts.METEOROLOGIST_BACKSTORY,
            verbose=True, # Requirement 2: Show traces in console
            allow_delegation=False,
            tools=[weather_tool],
            llm=llm
        )

        agronomist = Agent(
            role='Senior Agronomist',
            goal='Analyze weather for crop risks and recommend actions.',
            backstory=prompts.AGRONOMIST_BACKSTORY,
            verbose=True, # Requirement 2: Show traces in console
            allow_delegation=False,
            llm=llm
        )

        # ==========================================
        # TASKS
        # ==========================================

        task_geocode = Task(
            description=prompts.get_geocode_task_desc(location_input),
            expected_output=prompts.GEOCODE_TASK_EXPECTED,
            agent=geocoder
        )

        task_weather = Task(
            description=prompts.get_weather_task_desc(),
            expected_output=prompts.WEATHER_TASK_EXPECTED,
            agent=meteorologist
        )

        task_analysis = Task(
            description=prompts.get_analysis_task_desc(),
            expected_output=prompts.ANALYSIS_TASK_EXPECTED,
            agent=agronomist
        )

        # ==========================================
        # THE CREW
        # ==========================================

        crew = Crew(
            agents=[geocoder, meteorologist, agronomist],
            tasks=[task_geocode, task_weather, task_analysis],
            process=Process.sequential,
            verbose=True,
            memory=True, # Requirement 1: Conversational context/memory
            embedder={
                "provider": config['embedder']['provider'],
                "config": {
                    "model": config['embedder']['model_name'],
                    # "api_key": google_api_key
                }
            },
            output_log_file=log_file_path # Requirement 3: Timestamped log file
        )

        return crew.kickoff(inputs={'location_input': location_input})