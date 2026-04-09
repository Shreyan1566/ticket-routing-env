"""
Ticket Routing Environment — OpenEnv-compatible package.

Install from HuggingFace Space:
    pip install git+https://huggingface.co/spaces/Shreyan1567/ticket-routing-env

Usage:
    from ticket_routing_env import TicketRoutingEnv, TicketRoutingAction

    with TicketRoutingEnv(base_url="https://shreyan1567-ticket-routing-env.hf.space") as env:
        result = env.reset(task_id="easy_routing")
        result = env.step(department="billing", confidence=0.9)
"""

from models import TicketRoutingAction, TicketRoutingObservation, TicketRoutingState, StepResult, ResetResult
from client import TicketRoutingEnv

__all__ = [
    "TicketRoutingEnv",
    "TicketRoutingAction",
    "TicketRoutingObservation",
    "TicketRoutingState",
    "StepResult",
    "ResetResult",
]

__version__ = "1.0.0"
