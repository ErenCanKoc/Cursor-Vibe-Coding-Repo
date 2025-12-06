# System Overview

This document explains how the fan-out research demo works, how the pieces fit together, and what to know when extending or operating the system.

## Architecture
- **`fanout_ai.py`**: Core workflow that plans sub-queries, gathers context, and synthesizes GEO-ready answers via OpenAI structured outputs.
- **`app.py`**: Flask adapter that exposes the workflow through a web UI (`/`) and a JSON API (`/api/fanout`).
- **`templates/index.html`**: Single-page form to submit source content and keywords and view the generated answer blocks.

## Data models
`fanout_ai.py` uses Pydantic to validate model responses:
- `SubQueryPlan`: `{ main_intent: str, sub_queries: list[str] }`
- `AnswerBlock`: `{ heading: str, content: str, intent_category: Literal["Definition", "Comparison", "Limitations", "How-to"], source_quality_score: int, relevance_score: int }`
- `FanOutResult`: `{ analysis_summary: str, blocks: list[AnswerBlock] }`

These models power OpenAI's structured outputs to reduce parsing errors and enforce validation rules (length, categories, score ranges).

## Workflow
1. **Plan sub-queries (`step_1_plan_queries`)**
   - Prompts a planning model (`gpt-4o-2024-08-06`) to generate 3–5 distinct sub-queries for the target keyword.
   - Accepts optional source content to guide the question set but still forces intent diversity.
2. **Fetch context (`SearchTool.search_multiple`)**
   - A mock search layer that calls OpenAI (`gpt-4o`) to produce concise, synthetic snippets for each query.
   - Provides consistent structure without needing a real search API; logs at INFO level for observability.
3. **Synthesize answers (`step_3_synthesize`)**
   - Combines the keyword, plan, and search snippets into a synthesis prompt for `gpt-4o-2024-08-06`.
   - Returns a validated `FanOutResult` with 40–80 word, snippet-first answers and quality/relevance scores.
4. **Entrypoints**
   - `run_fan_out_workflow(keyword, content_text)`: full pipeline for internal use.
   - `run_tool(content_text, keyword)`: safe adapter used by Flask; returns `{ "result": ... }` or `{ "error": ... }`.

## API contract
`POST /api/fanout`
- **Request body**: `{ "content_text": "string", "keyword": "string" }` (keyword required).
- **Success response**: `{ "result": FanOutResult }`.
- **Error response**: `{ "error": "message" }` (e.g., missing keyword or upstream failure).

## Configuration and dependencies
- Requires `OPENAI_API_KEY` (dotenv-supported).
- Core dependencies: `openai`, `pydantic`, `python-dotenv`, `flask`.
- Logging is configured at INFO level in `fanout_ai.py` to trace mock search activity and exceptions.

## Operational notes
- The mock search uses OpenAI for deterministic structure; swap `SearchTool._mock_search` to integrate a real search API.
- Validation relies on the declared Pydantic models. Extend them if new fields are added to the prompts or UI.
- Flask runs in debug mode for local development; configure production settings separately if deploying.
