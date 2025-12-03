# Fan-Out Query Research Model

This repository provides a Python utility and FastAPI service for generating and validating GEO-focused fan-out query snippets using OpenAI's structured outputs.

## Features
- Builds GEO-compliant answer blocks (40-80 words, no ambiguous pronouns, validated intent categories, and relevance scores).
- Requests 3-5 unique fan-out queries per keyword and enforces per-block validation via Pydantic.
- Structured OpenAI chat completion request with dedicated system/user prompts for fan-out research.
- Simple CLI entry point or FastAPI service to run the model against sample content.

## Requirements
- Python 3.10+
- Dependencies: `openai`, `pydantic`, `python-dotenv`, `fastapi`, `uvicorn` (install via `pip install -r requirements.txt`).
- An `OPENAI_API_KEY` environment variable (set in a `.env` file or exported in your shell).

## Setup
1. Clone the repository and navigate into it:
   ```bash
   git clone <repo-url>
   cd Cursor-Vibe-Coding-Repo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your API key to a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

## Usage
Run the example end-to-end flow with the included sample content:
```bash
python fanout_ai.py
```

To integrate with your own content and keyword in another script, import the module directly:
```python
from fanout_ai import generate_geo_content

result = generate_geo_content(your_blog_text, "target keyword")
if result:
    for block in result.blocks:
        print(block.heading, block.content)
```

### Run as a FastAPI service
Launch the API server (default port 8000):
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Example request:
```bash
curl -X POST "http://localhost:8000/fanout" \
  -H "Content-Type: application/json" \
  -d '{"content_text": "<blog text>", "keyword": "online form builder"}'
```

Endpoints:
- `GET /health` — simple status check.
- `POST /fanout` — accepts `content_text` and `keyword`, returns validated `FanOutResult`.

## Validation Rules (Summary)
- **Intent categories** must be one of: `Tanım`, `Karşılaştırma`, `Kısıtlar`, `Nasıl Yapılır`.
- **Content length** must be 40-80 words and cannot start with ambiguous pronouns (e.g., "bu", "this", "they").
- **Headings** cannot be blank and should restate the target query.
- **Relevance scores** must be between 0 and 100.
- Each run must produce **3-5 unique** `target_query` values.

## Prompt Strategy
- The system prompt instructs the model to craft standalone GEO snippets, prioritize numeric details, and apply a light "Because/Therefore" reasoning style.
- The user prompt trims the source content (up to 4000 characters), requests 3-5 fan-out queries missing from the source snippet, and asks for concise, focused headings per block.

## Notes
- The script prints a clear error message if the model call fails and returns `None` instead of raising.
- Adjust the sample content and `keyword` in the `__main__` section to test different scenarios.
