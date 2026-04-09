"""
Ticket Routing Environment — OpenEnv-compatible server.
Agent reads customer support tickets and routes them to the correct department.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# Ticket data
# ─────────────────────────────────────────────

TICKETS: Dict[str, List[Dict[str, Any]]] = {
    "easy_routing": [
        {
            "ticket_id": "TKT-001",
            "sender": "alice@example.com",
            "priority": "high",
            "text": "My invoice for last month shows double charges. I was billed twice for the same subscription. Please refund the extra amount immediately.",
            "correct_department": "billing",
        },
        {
            "ticket_id": "TKT-002",
            "sender": "bob@example.com",
            "priority": "medium",
            "text": "The software keeps crashing every time I try to export a PDF. I've reinstalled it twice but the bug persists. Please fix this technical issue.",
            "correct_department": "technical",
        },
        {
            "ticket_id": "TKT-003",
            "sender": "carol@example.com",
            "priority": "low",
            "text": "I'm interested in upgrading my current plan to the enterprise tier. Can someone from your sales team contact me with pricing information?",
            "correct_department": "sales",
        },
        {
            "ticket_id": "TKT-004",
            "sender": "dave@example.com",
            "priority": "high",
            "text": "I need to report a workplace harassment incident involving my manager. This is urgent and confidential. Who should I contact in HR?",
            "correct_department": "hr",
        },
        {
            "ticket_id": "TKT-005",
            "sender": "eve@example.com",
            "priority": "high",
            "text": "We received a cease and desist letter and need immediate legal counsel. Please connect us with your legal department.",
            "correct_department": "legal",
        },
    ],
    "medium_routing": [
        {
            "ticket_id": "TKT-101",
            "sender": "frank@example.com",
            "priority": "medium",
            "text": "I was charged for a feature that stopped working three weeks ago. I'd like a refund for the period it wasn't functioning properly.",
            "correct_department": "billing",
        },
        {
            "ticket_id": "TKT-102",
            "sender": "grace@example.com",
            "priority": "medium",
            "text": "Our team is growing and the current setup isn't scaling well. The dashboard loads slowly and sometimes times out for large datasets.",
            "correct_department": "technical",
        },
        {
            "ticket_id": "TKT-103",
            "sender": "henry@example.com",
            "priority": "low",
            "text": "We're evaluating several vendors for our Q3 rollout. Could someone walk us through custom pricing for a 200-seat deployment?",
            "correct_department": "sales",
        },
        {
            "ticket_id": "TKT-104",
            "sender": "iris@example.com",
            "priority": "medium",
            "text": "One of our employees left the company last Friday. We need to ensure their access is revoked and understand the offboarding process.",
            "correct_department": "hr",
        },
        {
            "ticket_id": "TKT-105",
            "sender": "jack@example.com",
            "priority": "high",
            "text": "A competitor appears to be using our patented methodology in their product. We need to discuss our options for protecting our intellectual property.",
            "correct_department": "legal",
        },
    ],
    "hard_routing": [
        {
            "ticket_id": "TKT-201",
            "sender": "kate@example.com",
            "priority": "high",
            "text": "Our API integration broke after your last update and now our payments aren't processing. Customers are complaining and we're losing revenue.",
            "correct_department": "technical",
        },
        {
            "ticket_id": "TKT-202",
            "sender": "liam@example.com",
            "priority": "medium",
            "text": "We've been using the platform for 6 months and want to discuss renewal. The price seems high compared to competitors. Also, the export feature has been broken for two weeks.",
            "correct_department": "sales",
        },
        {
            "ticket_id": "TKT-203",
            "sender": "mia@example.com",
            "priority": "high",
            "text": "I received an unexpected charge and when I tried to dispute it through your portal, the page crashed. Now I can't access my account at all.",
            "correct_department": "billing",
        },
        {
            "ticket_id": "TKT-204",
            "sender": "noah@example.com",
            "priority": "high",
            "text": "An employee has been sharing client data with unauthorized third parties. We need to understand our liability and also take immediate disciplinary action.",
            "correct_department": "legal",
        },
        {
            "ticket_id": "TKT-205",
            "sender": "olivia@example.com",
            "priority": "medium",
            "text": "We recently merged with another company and need to consolidate two accounts. There are questions about user roles, access levels, and how billing will work across the merged entity.",
            "correct_department": "billing",
        },
    ],
}

DEPARTMENTS = ["billing", "technical", "sales", "hr", "legal"]

# Partial credit: related departments get 0.3 reward
PARTIAL_CREDIT: Dict[str, List[str]] = {
    "billing": ["sales"],
    "technical": [],
    "sales": ["billing"],
    "hr": ["legal"],
    "legal": ["hr"],
}

# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────


class Observation(BaseModel):
    ticket_id: str
    ticket_text: str
    sender: str
    priority: str
    current_step: int
    done: bool
    feedback: str
    score_so_far: float
    available_departments: List[str] = DEPARTMENTS


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class ResetResult(BaseModel):
    observation: Observation
    info: Dict[str, Any] = {}


class Action(BaseModel):
    department: str = Field(..., description="Target department")
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(None, description="Agent's reasoning")


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str
    ticket_count: int


class StateResponse(BaseModel):
    task_id: Optional[str]
    current_ticket: Optional[Dict[str, Any]]
    current_step: int
    total_reward: float
    done: bool
    session_id: str


# ─────────────────────────────────────────────
# Environment state
# ─────────────────────────────────────────────


class EnvState:
    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self.tickets: List[Dict[str, Any]] = []
        self.current_idx: int = 0
        self.current_step: int = 0
        self.total_reward: float = 0.0
        self.done: bool = False
        self.session_id: str = str(uuid.uuid4())
        self.max_steps: int = 5

    def current_ticket(self) -> Optional[Dict[str, Any]]:
        if self.current_idx < len(self.tickets):
            return self.tickets[self.current_idx]
        return None


_state = EnvState()

# ─────────────────────────────────────────────
# Grader
# ─────────────────────────────────────────────


def grade_action(action: Action, ticket: Dict[str, Any]) -> float:
    """
    Returns reward in [0.0, 1.0]:
      1.0 — correct department
      0.3 — partially correct (related department)
      0.0 — wrong
    Bonus +0.1 if correct AND confidence >= 0.7 (capped at 1.0).
    """
    chosen = action.department.lower().strip()
    correct = ticket["correct_department"]

    if chosen == correct:
        reward = 1.0
        if action.confidence >= 0.7:
            reward = min(1.0, reward + 0.1)
        return round(reward, 3)

    if chosen in PARTIAL_CREDIT.get(correct, []):
        return 0.3

    return 0.0


def build_feedback(action: Action, ticket: Dict[str, Any], reward: float) -> str:
    correct = ticket["correct_department"]
    chosen = action.department.lower().strip()
    if chosen == correct:
        return f"✓ Correct! '{chosen}' is the right department. Reward: {reward:.2f}"
    if reward > 0:
        return f"~ Partial credit. '{chosen}' is related but '{correct}' was the best fit. Reward: {reward:.2f}"
    return f"✗ Incorrect. '{chosen}' routed to wrong dept. Correct was '{correct}'. Reward: {reward:.2f}"


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="Ticket Routing Environment",
    description="OpenEnv-compatible environment for AI agent ticket routing.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "ticket-routing-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks() -> List[TaskInfo]:
    return [
        TaskInfo(
            id="easy_routing",
            name="Easy Ticket Routing",
            description="Route clearly-worded tickets with obvious department keywords.",
            difficulty="easy",
            ticket_count=len(TICKETS["easy_routing"]),
        ),
        TaskInfo(
            id="medium_routing",
            name="Medium Ticket Routing",
            description="Route moderately complex tickets requiring context understanding.",
            difficulty="medium",
            ticket_count=len(TICKETS["medium_routing"]),
        ),
        TaskInfo(
            id="hard_routing",
            name="Hard Ticket Routing",
            description="Route ambiguous, multi-topic tickets requiring nuanced judgment.",
            difficulty="hard",
            ticket_count=len(TICKETS["hard_routing"]),
        ),
    ]


@app.post("/reset")
def reset(body: Optional[Dict[str, Any]] = None) -> ResetResult:
    global _state
    _state = EnvState()

    # Default task if not specified
    task_id = "easy_routing"
    if body and body.get("task_id"):
        task_id = body["task_id"]

    if task_id not in TICKETS:
        task_id = "easy_routing"

    _state.task_id = task_id
    _state.tickets = list(TICKETS[task_id])  # copy
    _state.current_idx = 0
    _state.current_step = 0
    _state.total_reward = 0.0
    _state.done = False

    ticket = _state.current_ticket()
    obs = Observation(
        ticket_id=ticket["ticket_id"],
        ticket_text=ticket["text"],
        sender=ticket["sender"],
        priority=ticket["priority"],
        current_step=0,
        done=False,
        feedback="New episode started. Route the ticket to the correct department.",
        score_so_far=0.0,
    )
    return ResetResult(
        observation=obs,
        info={"task_id": task_id, "total_tickets": len(_state.tickets)},
    )


@app.post("/step")
def step(action: Action) -> StepResult:
    global _state

    if _state.done:
        ticket = _state.current_ticket() or {"ticket_id": "done", "text": "", "sender": "", "priority": ""}
        obs = Observation(
            ticket_id=ticket.get("ticket_id", "done"),
            ticket_text=ticket.get("text", ""),
            sender=ticket.get("sender", ""),
            priority=ticket.get("priority", ""),
            current_step=_state.current_step,
            done=True,
            feedback="Episode already finished. Call /reset to start again.",
            score_so_far=_state.total_reward,
        )
        return StepResult(observation=obs, reward=0.0, done=True)

    ticket = _state.current_ticket()
    if ticket is None:
        _state.done = True
        obs = Observation(
            ticket_id="none",
            ticket_text="",
            sender="",
            priority="",
            current_step=_state.current_step,
            done=True,
            feedback="No more tickets.",
            score_so_far=_state.total_reward,
        )
        return StepResult(observation=obs, reward=0.0, done=True)

    # Grade
    reward = grade_action(action, ticket)
    feedback = build_feedback(action, ticket, reward)

    _state.total_reward += reward
    _state.current_step += 1
    _state.current_idx += 1

    # Check done
    done = _state.current_idx >= len(_state.tickets) or _state.current_step >= _state.max_steps

    if done:
        _state.done = True
        next_ticket = ticket  # reuse last ticket for final obs
        final_score = _state.total_reward / len(_state.tickets)
        feedback += f" | Episode complete! Final score: {final_score:.3f}"
    else:
        next_ticket = _state.current_ticket()

    obs = Observation(
        ticket_id=next_ticket["ticket_id"],
        ticket_text=next_ticket["text"],
        sender=next_ticket["sender"],
        priority=next_ticket["priority"],
        current_step=_state.current_step,
        done=done,
        feedback=feedback,
        score_so_far=_state.total_reward,
    )

    return StepResult(
        observation=obs,
        reward=reward,
        done=done,
        info={
            "correct_department": ticket["correct_department"],
            "chosen_department": action.department,
            "step": _state.current_step,
        },
    )


@app.get("/state")
def state() -> StateResponse:
    ticket = _state.current_ticket()
    return StateResponse(
        task_id=_state.task_id,
        current_ticket=ticket,
        current_step=_state.current_step,
        total_reward=_state.total_reward,
        done=_state.done,
        session_id=_state.session_id,
    )
