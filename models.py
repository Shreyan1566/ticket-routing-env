"""
Typed Pydantic models for the Ticket Routing Environment.
Required for OpenEnv multi-mode deployment (pip-installable client).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


DEPARTMENTS = ["billing", "technical", "sales", "hr", "legal"]


class TicketRoutingAction(BaseModel):
    """Action: route a ticket to a department."""
    department: str = Field(..., description="Target department")
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(None, description="Agent reasoning")


class TicketRoutingObservation(BaseModel):
    """Observation returned after each step."""
    ticket_id: str
    ticket_text: str
    sender: str
    priority: str
    current_step: int
    done: bool
    feedback: str
    score_so_far: float
    available_departments: List[str] = DEPARTMENTS


class TicketRoutingState(BaseModel):
    """Full environment state."""
    task_id: Optional[str] = None
    current_step: int = 0
    total_reward: float = 0.0
    done: bool = False
    session_id: str = ""


class StepResult(BaseModel):
    """Result of a step action."""
    observation: TicketRoutingObservation
    reward: float
    done: bool
    info: dict = {}


class ResetResult(BaseModel):
    """Result of a reset call."""
    observation: TicketRoutingObservation
    info: dict = {}
