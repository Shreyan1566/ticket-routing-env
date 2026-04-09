"""
inference.py — Ticket Routing Environment baseline agent.

Structured stdout format (required by OpenEnv validator):
  [START] task=<name> env=<env> model=<model>
  [STEP]  step=<n> reward=<float> done=<bool> action=<str>
  [END]   task=<name> score=<float> success=<bool> steps=<n>

All prints use flush=True so the validator captures them immediately.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ─────────────────────────────────────────────
# Configuration — read from environment variables
# ─────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — required secret
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional: for from_docker_image()

# The HuggingFace Space URL where the environment is running
ENV_URL: str = os.environ.get(
    "ENV_URL",
    "https://shreyan1567-ticket-routing-env.hf.space",
).rstrip("/")

# Task IDs to evaluate
TASK_IDS: List[str] = ["easy_routing", "medium_routing", "hard_routing"]

MAX_STEPS: int = 5
SUCCESS_THRESHOLD: float = 0.6
DEPARTMENTS: List[str] = ["billing", "technical", "sales", "hr", "legal"]

# ─────────────────────────────────────────────
# Structured logging (REQUIRED — do not modify format)
# ─────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    """Emit the [START] block to stdout."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    reward: float,
    done: bool,
    action: str,
    error: Optional[str] = None,
) -> None:
    """Emit a [STEP] block to stdout."""
    line = f"[STEP] step={step} reward={reward:.4f} done={done} action={action}"
    if error:
        line += f" error={error}"
    print(line, flush=True)


def log_end(task: str, score: float, success: bool, steps: int) -> None:
    """Emit the [END] block to stdout."""
    print(
        f"[END] task={task} score={score:.4f} success={success} steps={steps}",
        flush=True,
    )


# ─────────────────────────────────────────────
# Environment client (plain httpx — no SDK needed)
# ─────────────────────────────────────────────


class TicketRoutingClient:
    """Simple HTTP client for the Ticket Routing environment."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def health(self) -> Dict[str, Any]:
        resp = self.client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def reset(self, task_id: str) -> Dict[str, Any]:
        resp = self.client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, department: str, confidence: float, reasoning: str) -> Dict[str, Any]:
        resp = self.client.post(
            f"{self.base_url}/step",
            json={
                "department": department,
                "confidence": confidence,
                "reasoning": reasoning,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def get_state(self) -> Dict[str, Any]:
        resp = self.client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self.client.close()


# ─────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert customer support ticket router.
Your job is to read a support ticket and decide which department should handle it.

Available departments:
- billing: payment issues, invoices, refunds, subscription charges, pricing disputes
- technical: software bugs, crashes, API issues, performance problems, integrations
- sales: upgrades, pricing inquiries, renewals, demos, enterprise deals
- hr: employee relations, onboarding, offboarding, workplace incidents, leave policies
- legal: contracts, intellectual property, compliance, lawsuits, regulatory matters

Respond ONLY with a valid JSON object. No extra text. Example:
{"department": "billing", "confidence": 0.95, "reasoning": "Ticket mentions double charge and refund request."}
"""


def get_agent_action(
    client: OpenAI,
    ticket_text: str,
    ticket_id: str,
    feedback: str,
    step: int,
) -> Dict[str, Any]:
    """Ask the LLM which department to route to. Returns dict with department/confidence/reasoning."""
    user_msg = (
        f"Ticket ID: {ticket_id}\n"
        f"Step: {step}\n"
        f"Previous feedback: {feedback}\n\n"
        f"Ticket content:\n{ticket_text}\n\n"
        f"Which department should handle this ticket? Respond with JSON only."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        dept = str(data.get("department", "technical")).lower().strip()
        if dept not in DEPARTMENTS:
            dept = "technical"
        confidence = float(data.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(data.get("reasoning", ""))
        return {"department": dept, "confidence": confidence, "reasoning": reasoning}

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # Fallback: simple keyword-based routing
        return _keyword_fallback(ticket_text)


def _keyword_fallback(ticket_text: str) -> Dict[str, Any]:
    """Rule-based fallback when LLM is unavailable."""
    text = ticket_text.lower()
    if any(w in text for w in ["invoice", "charge", "refund", "payment", "billing", "subscription fee"]):
        dept = "billing"
    elif any(w in text for w in ["bug", "crash", "error", "api", "technical", "performance", "broken", "not working"]):
        dept = "technical"
    elif any(w in text for w in ["upgrade", "pricing", "enterprise", "renewal", "demo", "sales"]):
        dept = "sales"
    elif any(w in text for w in ["employee", "harassment", "hr", "onboarding", "offboarding", "workplace"]):
        dept = "hr"
    elif any(w in text for w in ["legal", "lawsuit", "patent", "contract", "compliance", "cease", "liability"]):
        dept = "legal"
    else:
        dept = "technical"
    return {"department": dept, "confidence": 0.6, "reasoning": "Keyword-based fallback routing."}


# ─────────────────────────────────────────────
# Run one task episode
# ─────────────────────────────────────────────


def run_task(env: TicketRoutingClient, llm: OpenAI, task_id: str) -> float:
    """
    Run one full episode for a given task.
    Emits [START], [STEP]×n, [END] to stdout.
    Returns final score in [0.0, 1.0].
    """
    log_start(task=task_id, env="ticket-routing-env", model=MODEL_NAME)

    try:
        result = env.reset(task_id=task_id)
    except Exception as exc:
        print(f"[DEBUG] reset() failed: {exc}", flush=True)
        log_end(task=task_id, score=0.0, success=False, steps=0)
        return 0.0

    obs = result.get("observation", {})
    total_reward = 0.0
    steps_taken = 0
    done = obs.get("done", False)

    for step_num in range(1, MAX_STEPS + 1):
        if done:
            break

        ticket_text = obs.get("ticket_text", "")
        ticket_id = obs.get("ticket_id", "unknown")
        feedback = obs.get("feedback", "")

        # Get action from LLM (or fallback)
        action = get_agent_action(
            client=llm,
            ticket_text=ticket_text,
            ticket_id=ticket_id,
            feedback=feedback,
            step=step_num,
        )

        try:
            step_result = env.step(
                department=action["department"],
                confidence=action["confidence"],
                reasoning=action["reasoning"],
            )
            obs = step_result.get("observation", {})
            reward = float(step_result.get("reward", 0.0))
            done = step_result.get("done", False)
        except Exception as exc:
            print(f"[DEBUG] step() failed: {exc}", flush=True)
            reward = 0.0
            done = True

        total_reward += reward
        steps_taken = step_num

        log_step(
            step=step_num,
            reward=reward,
            done=done,
            action=action["department"],
        )

        if done:
            break

    # Score = mean reward per ticket (5 tickets per task)
    total_tickets = 5
    score = total_reward / total_tickets
    score = max(0.0, min(1.0, score))
    success = score >= SUCCESS_THRESHOLD

    log_end(task=task_id, score=score, success=success, steps=steps_taken)
    return score


# ─────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────


def main() -> None:
    print("[DEBUG] Starting inference.py", flush=True)
    print(f"[DEBUG] ENV_URL={ENV_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)

    # Set up clients
    env = TicketRoutingClient(base_url=ENV_URL)
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Verify environment is live
    try:
        health = env.health()
        print(f"[DEBUG] Environment health: {health}", flush=True)
    except Exception as exc:
        print(f"[DEBUG] Health check failed: {exc}", flush=True)
        print("[DEBUG] Continuing anyway — env may still work...", flush=True)

    # Run all three tasks
    all_scores: List[float] = []
    for task_id in TASK_IDS:
        print(f"\n[DEBUG] ===== Running task: {task_id} =====", flush=True)
        try:
            score = run_task(env=env, llm=llm, task_id=task_id)
            all_scores.append(score)
            print(f"[DEBUG] Task {task_id} score: {score:.4f}", flush=True)
        except Exception as exc:
            print(f"[DEBUG] Task {task_id} crashed: {exc}", flush=True)
            # Still emit [START] and [END] so validator sees them
            log_start(task=task_id, env="ticket-routing-env", model=MODEL_NAME)
            log_end(task=task_id, score=0.0, success=False, steps=0)
            all_scores.append(0.0)

        # Small delay between tasks
        time.sleep(1)

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n[DEBUG] All task scores: {all_scores}", flush=True)
    print(f"[DEBUG] Overall mean score: {overall:.4f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
