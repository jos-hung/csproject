"""Agent data models and registry for the AI management system."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime


class AgentStatus(Enum):
    """Lifecycle states of a managed AI agent."""

    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class AgentType(Enum):
    """Supported agent types."""

    GENERIC = "generic"
    CLASSIFIER = "classifier"
    SUMMARIZER = "summarizer"
    QA = "qa"
    CHAT = "chat"


@dataclass
class Agent:
    """Represents a managed AI agent."""

    name: str
    description: str
    agent_type: str = AgentType.GENERIC.value
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: AgentStatus = AgentStatus.CREATED
    created_at: datetime = field(default_factory=lambda: datetime.now())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the agent to a plain dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.agent_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class AgentRegistry:
    """In-memory registry that stores and looks up agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}

    def add(self, agent: Agent) -> None:
        self._agents[agent.id] = agent

    def remove(self, agent_id: str) -> Optional[Agent]:
        return self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Optional[Agent]:
        return self._agents.get(agent_id)

    def find_by_name(self, name: str) -> Optional[Agent]:
        for agent in self._agents.values():
            if agent.name == name:
                return agent
        return None

    def resolve(self, identifier: str) -> Optional[Agent]:
        """Look up an agent by ID first, then by name."""
        return self.get(identifier) or self.find_by_name(identifier)

    def all(self) -> list[Agent]:
        return list(self._agents.values())
