"""AI management system.

``ManagementSystem`` is the central controller.  It accepts either
programmatic calls (``create_agent``, ``start_agent``, …) or free-form
natural language commands that are first processed by the :class:`QueryAgent`
from :mod:`query` and then dispatched to the appropriate method.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents import Agent, AgentRegistry, AgentStatus, AgentType
from query import QueryAgent


class ManagementSystem:
    """Manages a collection of AI agents.

    The system exposes two layers of API:

    1. **Direct API** – call :meth:`create_agent`, :meth:`start_agent`, etc.
       directly with typed arguments.
    2. **NL API** – call :meth:`process_command` with a free-text string; the
       embedded :class:`QueryAgent` translates it into one of the direct-API
       calls and returns the result.

    Args:
        query_agent: Optional pre-configured :class:`QueryAgent`.  A default
                     instance (using ``OPENAI_API_KEY`` from the environment)
                     is created when *None*.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, query_agent: Optional[QueryAgent] = None) -> None:
        self._registry = AgentRegistry()
        self._query_agent = query_agent or QueryAgent()

    # ------------------------------------------------------------------
    # NL command gateway
    # ------------------------------------------------------------------

    def process_command(self, command: str) -> Dict[str, Any]:
        """Interpret *command* with the AI agent and execute the action.

        The :class:`QueryAgent` returns a structured dict; this method
        dispatches to the correct management method and returns its result
        enriched with the AI's human-readable ``message``.
        """
        parsed = self._query_agent.query(command)
        action = parsed.get("action", "unknown")
        params = parsed.get("parameters", {})
        ai_message = parsed.get("message", "")

        result = self._dispatch(action, params)
        result["ai_message"] = ai_message
        return result

    # ------------------------------------------------------------------
    # Direct management API
    # ------------------------------------------------------------------

    def create_agent(
        self,
        name: str,
        description: str = "",
        agent_type: str = AgentType.GENERIC.value,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a new AI agent and return its serialised representation."""
        agent = Agent(
            name=name,
            description=description,
            agent_type=agent_type,
            metadata=metadata or {},
        )
        self._registry.add(agent)
        return {"status": "success", "action": "created", "agent": agent.to_dict()}

    def delete_agent(self, identifier: str) -> Dict[str, Any]:
        """Remove an agent from the registry by ID or name."""
        agent = self._registry.resolve(identifier)
        if agent is None:
            return self._not_found(identifier)
        self._registry.remove(agent.id)
        return {"status": "success", "action": "deleted", "agent_id": agent.id, "name": agent.name}

    def start_agent(self, identifier: str) -> Dict[str, Any]:
        """Transition an agent to the RUNNING state."""
        agent = self._registry.resolve(identifier)
        if agent is None:
            return self._not_found(identifier)
        agent.status = AgentStatus.RUNNING
        return {"status": "success", "action": "started", "agent": agent.to_dict()}

    def stop_agent(self, identifier: str) -> Dict[str, Any]:
        """Transition an agent to the STOPPED state."""
        agent = self._registry.resolve(identifier)
        if agent is None:
            return self._not_found(identifier)
        agent.status = AgentStatus.STOPPED
        return {"status": "success", "action": "stopped", "agent": agent.to_dict()}

    def list_agents(self) -> Dict[str, Any]:
        """Return all registered agents."""
        return {
            "status": "success",
            "agents": [a.to_dict() for a in self._registry.all()],
            "count": len(self._registry.all()),
        }

    def get_status(self, identifier: str) -> Dict[str, Any]:
        """Return the current state of a single agent."""
        agent = self._registry.resolve(identifier)
        if agent is None:
            return self._not_found(identifier)
        return {"status": "success", "agent": agent.to_dict()}

    def query_agent(self, identifier: str, query: str) -> Dict[str, Any]:
        """Forward *query* to the AI query agent on behalf of a managed agent.

        This lets the management system ask the LLM agent something *about*
        a specific managed agent (e.g. "explain what this agent does").
        """
        agent = self._registry.resolve(identifier)
        if agent is None:
            return self._not_found(identifier)

        enriched = f"[Agent context: {agent.to_dict()}]\n{query}"
        parsed = self._query_agent.query(enriched)
        return {
            "status": "success",
            "agent_id": agent.id,
            "query": query,
            "response": parsed.get("message", str(parsed)),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route an action name to the corresponding management method."""
        identifier = params.get("id") or params.get("name", "")

        dispatch_table = {
            "create_agent": lambda: self.create_agent(
                name=params.get("name", "unnamed"),
                description=params.get("description", ""),
                agent_type=params.get("type", AgentType.GENERIC.value),
            ),
            "delete_agent": lambda: self.delete_agent(identifier),
            "start_agent": lambda: self.start_agent(identifier),
            "stop_agent": lambda: self.stop_agent(identifier),
            "list_agents": lambda: self.list_agents(),
            "get_status": lambda: self.get_status(identifier),
            "query_agent": lambda: self.query_agent(
                identifier, params.get("query", "")
            ),
        }

        handler = dispatch_table.get(action)
        if handler is None:
            return {
                "status": "unknown_action",
                "action": action,
                "message": f"Unsupported action: '{action}'",
            }
        return handler()

    @staticmethod
    def _not_found(identifier: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "message": f"Agent not found: '{identifier}'",
        }
