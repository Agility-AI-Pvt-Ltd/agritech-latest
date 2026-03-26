# Kisan Mitra Codebase Architecture

Generated from the current code layout.

## Main idea

This repository contains two different application flows:

1. `POST /api/advisory` and `POST /api/advisory/predefined`
   These use a simpler service pipeline:
   request validation -> crop stage -> weather/context lookup -> two-step advisory generation -> DB logging.

2. `POST /api/chat`
   This uses a LangGraph agent loop:
   load conversation state -> run a safety gate -> build prompt with profile + summary + recent turns -> let the LLM call tools -> loop until it can answer -> persist updated state/profile.

## Important architectural split

- `core/database.py` + `models/` + `repositories/`
  This is the async SQLAlchemy stack used by the traditional advisory endpoints.

- `pipeline/database/*`
  This is a separate sync persistence layer used by the chat agent.
  It talks to:
  - PostgreSQL for `user_profiles` and `conversation_states`
  - Redis for active conversation-state caching

That means the repo currently has two persistence approaches in parallel.

## Key runtime files

- `main.py`
  Startup lifecycle, DB init, retrieval checks, warm chat resources.

- `api/routes.py`
  All HTTP entrypoints.

- `services/advisory.py`
  Two-node LangGraph pipeline for non-chat advisory generation.

- `pipeline/graph.py`
  Chat orchestration, memory loading, safety screening, summarization, profile extraction, persistence.

- `pipeline/agent.py`
  Single-node LangGraph agent loop with tool calls.

- `pipeline/prompts/safety_prompt.py`
  Prompt used by the lightweight safety classifier before the agent runs.

- `pipeline/tools/dispatcher.py`
  Central tool router for the chat agent.

## Safety gate

The `/api/chat` graph now starts with a dedicated safety node.

- It first applies heuristic checks for obvious prompt injection, secret extraction, malware, exploit, or shell-command probing.
- Then it can call a smaller dedicated safety LLM to classify ambiguous queries.
- If a query is blocked, the graph ends early and returns a friendly Hindi rejection message.
- Blocked turns do not reach the main tool-capable agent node.

Relevant config:

- `CHAT_SAFETY_ENABLED`
- `CHAT_SAFETY_FAIL_CLOSED`
- `SAFETY_LLM_PROVIDER`
- `SAFETY_LLM_MODEL`
- `SAFETY_LLM_TEMPERATURE`

## Diagram

The SVG diagram is here:

- `docs/codebase_architecture.svg`
