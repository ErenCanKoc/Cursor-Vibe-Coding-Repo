# Fan-Out Query Research Model

This repository provides a Flask-based demo and reusable Python module for generating GEO-focused fan-out query snippets using OpenAI structured outputs. It plans sub-queries for a keyword, gathers synthetic search context, and synthesizes 40–80 word answer blocks that are ready for AI surfaces.

## Features
- End-to-end workflow that plans 3–5 sub-queries, gathers context, and synthesizes GEO-ready answers.
- Pydantic data models (`SubQueryPlan`, `AnswerBlock`, `FanOutResult`) to validate model responses.
- Flask UI and JSON API for interactive testing or programmatic use.
- Structured OpenAI chat completion requests with dedicated system/user prompts.

## Requirements
- Python 3.10+
- Dependencies: `openai`, `pydantic`, `python-dotenv`, `flask` (install via `pip install -r requirements.txt`).
- `OPENAI_API_KEY` environment variable (set in a `.env` file or exported in your shell).

## Setup
1. Clone the repository and navigate into it:
   ```bash
   git clone <repo-url>
   cd Cursor-Vibe-Coding-Repo
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your API key to a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

## Running the Flask app
Start the local server (the template folder is already configured in `app.py`):
```bash
python app.py
```
Then open http://127.0.0.1:5000 to use the form UI. Submitted content and keywords will run through the fan-out workflow and display the generated analysis summary plus answer blocks.

### JSON API
You can also hit the API directly:
```bash
curl -X POST http://127.0.0.1:5000/api/fanout \
  -H "Content-Type: application/json" \
  -d '{"content_text": "Your source text", "keyword": "electric vehicle range"}'
```
The response contains either a `result` payload matching the `FanOutResult` schema or an `error` message.

## Using the module directly
If you want to call the workflow in your own scripts without Flask:
```python
from fanout_ai import run_tool

content = "Short excerpt about battery safety standards..."
keyword = "lithium battery safety"
response = run_tool(content_text=content, keyword=keyword)

if "result" in response:
    for block in response["result"]["blocks"]:
        print(block["heading"], block["content"])
else:
    print("Error:", response["error"])
```

## Validation rules (summary)
- Intent categories must be one of `Definition`, `Comparison`, `Limitations`, `How-to`.
- Content length must be 40–80 words and cannot start with ambiguous pronouns.
- Headings cannot be blank and should restate the target query.
- Relevance and source quality scores are integers between 0 and 100.
- Each run should produce **3–5 unique** sub-queries and matching answer blocks.

## Prompt strategy
- The system prompt instructs the model to craft standalone GEO snippets with causal explanations and snippet-first structure.
- The user prompt provides the keyword, strategy, and mock search context for synthesis.
- The planning step uses its own prompt to create diverse, answerable sub-queries before synthesis.

## System documentation
See [`docs/system_overview.md`](docs/system_overview.md) for a deeper look at the architecture, data contracts, and operational notes.
