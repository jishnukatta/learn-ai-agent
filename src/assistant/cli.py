"""cli.py — the entry point.

Stage 6: run the explicit LangGraph version (agent_graph). Same behavior, but
the agent is now a graph we built ourselves (nodes + edges + state).
"""

from assistant.agent_graph import run_agent


def main() -> None:
    """Runs when you execute `uv run assistant`."""
    question = "What time is it right now, and what is 17 * 23?"
    print(f"USER: {question}\n")

    answer = run_agent(question)

    print("==================== FINAL ANSWER ====================")
    print(answer)


if __name__ == "__main__":
    main()
