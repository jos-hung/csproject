"""Real AI agent that provides LLM-powered services for AgentManagementLayer.

QueryAgent is injected into AgentManagementLayer and upgrades two steps:
  - detect_intent  : uses an LLM to classify the user's intent instead of
                     a simple keyword lookup.
  - aggregate_results : uses an LLM to produce a natural-language summary
                        of what the agents did, instead of a plain count.

Usage
-----
    from query import QueryAgent
    from management import AgentManagementLayer, AIAgent

    qa = QueryAgent()          # reads OPENAI_API_KEY from the environment
    manager = AgentManagementLayer(agents=[...], query_agent=qa)
    result  = manager.handle_request(request)
"""

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Supported intents (must match AgentManagementLayer.plan_actions keys)
# ---------------------------------------------------------------------------

_VALID_INTENTS = {"data_collection", "simulation", "report_generation", "general_task"}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_INTENT_SYSTEM_PROMPT = """
You are an intent classifier for an AI agent management system.
Classify the user's request into exactly one of the following intents:
  - data_collection   : collecting data, recording, building a dataset
  - simulation        : running simulations or virtual scenarios
  - report_generation : generating reports or summaries from existing data
  - general_task      : anything that does not fit the above categories

Respond with ONLY the intent label, nothing else.
""".strip()

_AGGREGATE_SYSTEM_PROMPT = """
You are an assistant that summarizes the results of a multi-agent workflow.
Given a JSON list of task results, write a concise, friendly human-readable
summary of what was accomplished. Keep it to 2-3 sentences.
""".strip()


# ---------------------------------------------------------------------------
# QueryAgent
# ---------------------------------------------------------------------------


class QueryAgent:
    """LLM-powered agent used by AgentManagementLayer.

    Args:
        model:    OpenAI chat model (default: ``gpt-4o-mini``).
        api_key:  OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Optional custom base URL for OpenAI-compatible APIs.
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

    # ------------------------------------------------------------------
    # Public API called by AgentManagementLayer
    # ------------------------------------------------------------------

    def detect_intent(self, text: str) -> str:
        """Classify *text* into one of the management system's intents.

        Falls back to ``"general_task"`` when the model returns an
        unrecognised label.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=20,
            temperature=0,
        )
        label = (response.choices[0].message.content or "").strip().lower()
        return label if label in _VALID_INTENTS else "general_task"

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a natural-language summary of *results* using an LLM."""
        import json

        results_json = json.dumps(results, indent=2)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _AGGREGATE_SYSTEM_PROMPT},
                {"role": "user", "content": results_json},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        summary = (response.choices[0].message.content or "").strip()

        successful = [r for r in results if r.get("status") == "success"]
        return {
            "status": "completed",
            "summary": summary,
            "details": successful,
        }

