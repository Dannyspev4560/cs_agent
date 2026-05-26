"""CLI entry point for the Customer Service Data Analyst Agent.

Usage:
    python main.py                        # default session
    python main.py --session my_session   # named session (restores history in Task 2)
"""

import argparse
import os
import sqlite3

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import build_graph

_DB_PATH = os.path.join(os.path.dirname(__file__), "agent", "storage", "checkpoints.db")


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer Service Data Analyst Agent")
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID — reuse the same ID to restore conversation history.",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": args.session}}

    print("=" * 60)
    print("  Customer Service Data Analyst Agent")
    print(f"  Session: {args.session}")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print()
        last_state = None
        try:
            for state in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values",
            ):
                last_state = state

            # Your idea: final answer is always the last message in the last state
            if last_state:
                print(f"\nAgent: {last_state['messages'][-1].content}")

        except Exception as exc:
            print(f"[Error] {exc}")


if __name__ == "__main__":
    main()
