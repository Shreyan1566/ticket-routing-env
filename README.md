# 🎫 Ticket Routing Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent learns to route customer support tickets to the correct department.

## Overview

Real-world task: customer support teams receive hundreds of tickets daily. Misrouting wastes time and frustrates customers. This environment trains an agent to accurately classify and route tickets across 5 departments.

| Department | Examples |
|------------|---------|
| `billing` | Invoices, refunds, charge disputes |
| `technical` | Bugs, crashes, API issues |
| `sales` | Upgrades, renewals, pricing |
| `hr` | Employee incidents, onboarding |
| `legal` | Contracts, IP, compliance |

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `easy_routing` | Easy | Clear, keyword-rich tickets |
| `medium_routing` | Medium | Context-dependent routing |
| `hard_routing` | Hard | Ambiguous, multi-topic tickets |

## Action Space

```json
{
  "department": "billing",     // one of: billing, technical, sales, hr, legal
  "confidence": 0.9,           // float in [0.0, 1.0]
  "reasoning": "Invoice issue" // optional string
}
```

## Observation Space

```json
{
  "ticket_id": "TKT-001",
  "ticket_text": "I was billed twice...",
  "sender": "alice@example.com",
  "priority": "high",
  "current_step": 1,
  "done": false,
  "feedback": "Correct! billing is right.",
  "score_so_far": 1.0,
  "available_departments": ["billing", "technical", "sales", "hr", "legal"]
}
```

## Reward Function

- **1.0** — Correct department
- **+0.1 bonus** — Correct AND confidence ≥ 0.7 (capped at 1.0)
- **0.3** — Partially correct (related department)
- **0.0** — Wrong department

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take an action |
| GET | `/state` | Get current state |

### Reset

```bash
curl -X POST https://your-space.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_routing"}'
```

### Step

```bash
curl -X POST https://your-space.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"department": "billing", "confidence": 0.9, "reasoning": "Refund request"}'
```

## Setup & Running Locally

```bash
git clone https://github.com/Shreyan1566/ticket-routing-env
cd ticket-routing-env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Running with Docker

```bash
docker build -t ticket-routing-env .
docker run -p 7860:7860 ticket-routing-env
```

## Running Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_api_key_here"
export ENV_URL="https://shreyan1567-ticket-routing-env.hf.space"

python inference.py
```

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | Your HuggingFace / OpenAI API key |
| `ENV_URL` | Your HuggingFace Space URL |

## Stdout Format

The inference script emits structured logs required by the OpenEnv validator:

```
[START] task=easy_routing env=ticket-routing-env model=gpt-4o-mini
[STEP] step=1 reward=1.0000 done=false action=billing
[STEP] step=2 reward=1.0000 done=false action=technical
[STEP] step=3 reward=0.3000 done=false action=sales
[STEP] step=4 reward=1.0000 done=false action=hr
[STEP] step=5 reward=1.0000 done=true action=legal
[END] task=easy_routing score=0.8600 success=true steps=5
```

## Team

**NullSec** — OpenEnv Hackathon 2026  
Shreyan Cheekoti & Anirudh Balla
