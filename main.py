"""Command-line interface for the AI management system.

Usage
-----
    python main.py                  # interactive REPL
    python main.py "list agents"    # single command from argv

Environment
-----------
    OPENAI_API_KEY   Required – passed through to the QueryAgent.
"""

import json
import sys

from management import ManagementSystem


def _pretty(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def run_once(system: ManagementSystem, command: str) -> None:
    print(f"\n> {command}")
    result = system.process_command(command)
    print(_pretty(result))


def repl(system: ManagementSystem) -> None:
    print("AI Management System – type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            command = input("management> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not command:
            continue
        if command.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            result = system.process_command(command)
            print(_pretty(result))
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}")


def main() -> None:
    system = ManagementSystem()

    if len(sys.argv) > 1:
        run_once(system, " ".join(sys.argv[1:]))
    else:
        repl(system)


if __name__ == "__main__":
    main()
