"""
HTTP client for the Ticket Routing Environment.
Required for OpenEnv multi-mode deployment (pip-installable client).

Usage:
    from ticket_routing_env.client import TicketRoutingEnv

    env = TicketRoutingEnv(base_url="https://shreyan1567-ticket-routing-env.hf.space")
    result = env.reset(task_id="easy_routing")
    result = env.step(department="billing", confidence=0.9)
    env.close()
"""

from typing import Any, Dict, Optional
import httpx

from models import TicketRoutingAction, ResetResult, StepResult, TicketRoutingState


class TicketRoutingEnv:
    """Synchronous HTTP client for the Ticket Routing Environment."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def reset(self, task_id: str = "easy_routing") -> ResetResult:
        resp = self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        return ResetResult(**resp.json())

    def step(
        self,
        department: str,
        confidence: float = 0.8,
        reasoning: Optional[str] = None,
    ) -> StepResult:
        action = TicketRoutingAction(
            department=department,
            confidence=confidence,
            reasoning=reasoning,
        )
        resp = self._client.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
        )
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> TicketRoutingState:
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return TicketRoutingState(**resp.json())

    def health(self) -> Dict[str, Any]:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
