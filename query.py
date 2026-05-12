"""AI agent that processes natural language management commands via an LLM.

The QueryAgent sends user commands to an OpenAI-compatible chat model and
expects a JSON response describing which management action to perform.  A
configurable system prompt keeps the agent focused on the management domain.
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are an AI agent management assistant embedded in an AI management system.
Your role is to interpret natural language commands and translate them into
structured management actions for other AI agents.

Always respond with a valid JSON object that has the following fields:
  - "action"     : one of create_agent | delete_agent | start_agent |
                   stop_agent | list_agents | get_status | query_agent |
                   unknown
  - "parameters" : an object with action-specific fields (see below)
  - "message"    : a concise human-readable description of what you will do

Parameter schemas per action
─────────────────────────────
create_agent  → { name, description?, type? }
delete_agent  → { id? , name? }
start_agent   → { id? , name? }
stop_agent    → { id? , name? }
get_status    → { id? , name? }
list_agents   → {}
query_agent   → { id? , name? , query }
unknown       → {}

Use "unknown" when the command does not map to any supported action.
""".strip()


# ---------------------------------------------------------------------------
# QueryAgent
# ---------------------------------------------------------------------------


class QueryAgent:
    """Translates natural language commands into management actions using an LLM.

    Args:
        model:      OpenAI model name (default: ``gpt-4o-mini``).
        api_key:    OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
                    environment variable when *None*.
        base_url:   Optional custom base URL for OpenAI-compatible APIs.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self._history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, user_input: str) -> Dict[str, Any]:
        """Send *user_input* to the LLM and return a structured action dict.

        The returned dictionary always contains at least the keys
        ``action``, ``parameters``, and ``message``.

        Raises:
            ValueError: If the model returns a response that is not valid JSON.
            openai.OpenAIError: On API-level errors.
        """
        self._history.append({"role": "user", "content": user_input})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                *self._history,
            ],
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        try:
            result: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned non-JSON response: {raw!r}"
            ) from exc

        self._history.append({"role": "assistant", "content": raw})
        return self._normalise(result)

    def reset(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure required top-level keys are always present."""
        result.setdefault("action", "unknown")
        result.setdefault("parameters", {})
        result.setdefault("message", "")
        return result
