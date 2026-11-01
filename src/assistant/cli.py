"""cli.py — the entry point.

Stage 2: run the AGENT LOOP. We ask a question that needs tools, so you can
watch the model decide to call them, see the results fed back, and get a final
answer.
"""

from assistant.agent import run_agent


def main() -> None:
    """Runs when you execute `uv run assistant`."""
    question = "What time is it right now, and what is 17 * 23?"
    print(f"USER: {question}")

    answer = run_agent(question, verbose=True)

    print("\n==================== FINAL ANSWER ====================")
    print(answer)


if __name__ == "__main__":
    main()
