"""Command-line interface for the AI management system.

Usage
-----
    python main.py                  # interactive REPL (requires OPENAI_API_KEY)
    python main.py --no-ai          # interactive REPL with keyword-based intent only

Environment
-----------
    OPENAI_API_KEY   Required when using the real AI agent (QueryAgent).
"""

import sys
import json

from management import AgentManagementLayer, AIAgent, UserRequest


def _build_manager(use_ai: bool) -> AgentManagementLayer:
    agents = [
        AIAgent(agent_id="AI Agent #1", profile=["action1", "action2", "action3", "action4", "action5"]),
        AIAgent(agent_id="AI Agent #2", profile=["action4", "action5", "action6"]),
        AIAgent(agent_id="AI Agent #3", profile=["action8", "action9", "action10"]),
    ]

    if use_ai:
        from query import QueryAgent
        query_agent = QueryAgent()
    else:
        query_agent = None

    return AgentManagementLayer(agents=agents, query_agent=query_agent)


def _pretty(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def repl(manager: AgentManagementLayer) -> None:
    mode = "AI (LLM)" if manager.query_agent else "keyword"
    print(f"AI Management System – intent detection mode: {mode}")
    print("Type your request and press Enter. Type 'exit' to quit.\n")

    user_id = "user_cli"
    location_id = "location_01"

    while True:
        try:
            raw = input("management> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        request = UserRequest(
            user_id=user_id,
            raw_input=raw,
            input_type="chat",
            location_id=location_id,
        )

        try:
            result = manager.handle_request(request)
            print(_pretty(result))
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}")


def main() -> None:
    use_ai = "--no-ai" not in sys.argv
    manager = _build_manager(use_ai)
    repl(manager)


if __name__ == "__main__":
    main()
