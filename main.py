import sys
import traceback

from src.agent import run_agri_crew
from src.prompts import RiskReport

def print_header():
    print("="*60)
    print(" 🌾 Welcome to the Agri-Weather Forecaster AI 🌾")
    print("="*60)
    print("Type a region, city, or address to get a 14-day agronomic")
    print("risk assessment. (Type 'exit' or 'quit' to close)")
    print("-" * 60)

def main():
    print_header()

    while True:
        try:
            # 1. Get User Input
            location_input = input("\n📍 Enter farming region (e.g., 'Bordeaux, France'): ").strip()
            
            if location_input.lower() in ['exit', 'quit']:
                print("\nExiting the forecaster. Goodbye! 👋")
                sys.exit(0)
                
            if not location_input:
                print("Please enter a valid location.")
                continue

            print(f"\n🚀 Initializing multi-agent crew for: '{location_input}'...")
            print("Watch the console for agent execution traces.\n")

            # 2 + 3. Build the Crew (with MCP-served tools) and kick off the workflow.
            # Tool calls flow through a FastMCP server launched as a subprocess.
            result = run_agri_crew(location_input)

            # 4. Display the Final Result
            print("\n" + "="*60)
            print(" 📋 FINAL AGRONOMIC REPORT ")
            print("="*60)
            _render_result(result)
            print("="*60)
            print("📂 Note: A detailed trace of this session has been saved in the 'logs/' folder.")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Goodbye! 👋")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ {type(e).__name__} during execution: {e}")
            print("Check your API keys, internet connection, or quota limits.")
            traceback.print_exc()


def _render_result(result):
    """Pretty-print a CrewAI result that may carry a Pydantic RiskReport."""
    report = getattr(result, "pydantic", None)
    if isinstance(report, RiskReport):
        print(f"Severity      : {report.severity.upper()}")
        print(f"Primary Risk  : {report.primary_risk}")
        if report.affected_dates:
            print(f"Affected Dates: {', '.join(report.affected_dates)}")
        print(f"Recommendation: {report.recommendation}")
        print("-" * 60)
        print(report.summary)
    else:
        print(result)

if __name__ == "__main__":
    # Ensure the script runs only when executed directly
    main()