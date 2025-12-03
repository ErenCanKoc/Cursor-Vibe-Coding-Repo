# Fan-Out Query Research Model

This repository provides a Python utility for generating and validating GEO-focused fan-out query snippets using OpenAI's structured outputs.

## Features
- Builds GEO-compliant answer blocks (40-80 words, no ambiguous pronouns, validated intent categories, and relevance scores).
- Requests 3-5 unique fan-out queries per keyword and enforces per-block validation via Pydantic.
- Structured OpenAI chat completion request with dedicated system/user prompts for fan-out research.
- Simple CLI entry point to run the model against sample content.

## Requirements
- Python 3.10+
- Dependencies: `openai`, `pydantic`, `python-dotenv` (install via `pip install -r requirements.txt` or `pip install openai pydantic python-dotenv`).
- An `OPENAI_API_KEY` environment variable (set in a `.env` file or exported in your shell).

## Setup
1. Clone the repository and navigate into it:
   ```bash
   git clone <repo-url>
   cd Cursor-Vibe-Coding-Repo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # or: pip install openai pydantic python-dotenv
   ```
3. Add your API key to a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

## Usage
Run the example end-to-end flow with the included sample content:
```bash
python "FanOut AI"
```

To integrate with your own content and keyword in another script, either rename the file to a valid module name (e.g., `fanout.py`) or load it dynamically:
```python
import importlib.util
from pathlib import Path

module_path = Path("FanOut AI").resolve()
spec = importlib.util.spec_from_file_location("fanout_ai", module_path)
fanout_ai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fanout_ai)

result = fanout_ai.generate_geo_content(your_blog_text, "target keyword")
if result:
    for block in result.blocks:
        print(block.heading, block.content)
```

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
