import argparse
import traceback

# Both backends expose: run(location_input: str) -> RiskReport-like result.
# We resolve the backend lazily so missing optional deps in one stack don't
# block usage of the other.

def _load_backend(framework: str):
    if framework == "crewai":
        from src_crewai.agent import run_agri_crew
        from src_crewai.prompts import RiskReport
        return run_agri_crew, RiskReport, None
    if framework == "langgraph":
        from src_langgraph.agent import run_agri_graph, shutdown
        from src_langgraph.prompts import RiskReport
        return run_agri_graph, RiskReport, shutdown
    raise ValueError(f"Unknown framework: {framework!r}")


def print_header(framework: str):
    print("="*60)
    print(" 🌾 Welcome to the Agri-Weather Forecaster AI 🌾")
    print("="*60)
    print(f"Backend: {framework}")
    print("Type a region, city, or address to get a 14-day agronomic")
    print("risk assessment. (Type 'exit' or 'quit' to close)")
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Agri-Weather Forecaster")
    parser.add_argument(
        "--framework",
        choices=["crewai", "langgraph"],
        default="crewai",
        help="Which agent orchestration backend to use (default: crewai).",
    )
    args = parser.parse_args()

    run_fn, RiskReport, shutdown_fn = _load_backend(args.framework)
    print_header(args.framework)

    try:
        _repl(run_fn, RiskReport)
    finally:
        if shutdown_fn is not None:
            shutdown_fn()


def _repl(run_fn, RiskReport):
    while True:
        try:
            # 1. Get User Input
            location_input = input("\n📍 Enter farming region (e.g., 'Bordeaux, France'): ").strip()

            if location_input.lower() in ['exit', 'quit']:
                print("\nExiting the forecaster. Goodbye! 👋")
                return

            if not location_input:
                print("Please enter a valid location.")
                continue

            print(f"\n🚀 Initializing multi-agent workflow for: '{location_input}'...")
            print("Watch the console for agent execution traces.\n")

            result = run_fn(location_input)

            # 4. Display the Final Result
            print("\n" + "="*60)
            print(" 📋 FINAL AGRONOMIC REPORT ")
            print("="*60)
            _render_result(result, RiskReport)
            print("="*60)
            print("📂 Note: A detailed trace of this session has been saved in the 'logs/' folder.")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Goodbye! 👋")
            return
        except Exception as e:
            print(f"\n❌ {type(e).__name__} during execution: {e}")
            print("Check your API keys, internet connection, or quota limits.")
            traceback.print_exc()


def _render_result(result, RiskReport):
    """Pretty-print a RiskReport or a CrewAI wrapper carrying one."""
    report = result if isinstance(result, RiskReport) else getattr(result, "pydantic", None)
    if report is None or not isinstance(report, RiskReport):
        print(result)
        return
    print(f"Severity      : {report.severity.upper()}")
    print(f"Primary Risk  : {report.primary_risk}")
    if report.affected_dates:
        print(f"Affected Dates: {', '.join(report.affected_dates)}")
    print(f"Recommendation: {report.recommendation}")
    print("-" * 60)
    print(report.summary)

if __name__ == "__main__":
    # Ensure the script runs only when executed directly
    main()