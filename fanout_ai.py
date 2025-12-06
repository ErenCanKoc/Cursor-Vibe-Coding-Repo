import logging
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Load API key from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. Validation Layer (GEO Rules) ---
# We enforce the GEO document rules (40-80 words, no pronouns) here via code.

_INTENT_CHOICES = {
    "Definition",
    "Comparison",
    "Limitations",
    "How-to",
}


class AnswerBlock(BaseModel):
    intent_category: str = Field(
        ...,
        description="The intent of the query: 'Definition', 'Comparison', 'Limitations', or 'How-to'",
    )
    target_query: str = Field(
        ..., 
        description="The likely user fan-out query (e.g., 'Jotform vs Zapier pricing')"
    )
    heading: str = Field(
        ..., 
        description="The heading to be used as H2 or H3 in the blog post"
    )
    content: str = Field(
        ..., 
        description="The snippet text containing the direct answer"
    )
    relevance_score: int = Field(
        ..., 
        description="LMP Relevance Score (0-100). How relevant is the content to the question?"
    )

    @field_validator("intent_category")
    def validate_intent(cls, value: str) -> str:
        if value not in _INTENT_CHOICES:
            raise ValueError(f"intent_category '{value}' is not supported. Valid options: {_INTENT_CHOICES}")
        return value

    @field_validator("content")
    def validate_geo_rules(cls, value: str) -> str:
        # Rule 1: Word Count (GEO Tactics: 40-80 words is the ideal snippet)
        word_count = len(value.split())
        if word_count < 40 or word_count > 80:
            raise ValueError(
                f"Answer Block length does not meet GEO standards ({word_count} words). Must be between 40-80."
            )

        # Rule 2: Ambiguous Pronouns (GEO Tactics: 'This', 'It' etc. are forbidden at the start)
        # Updated for English pronouns
        forbidden_starts = ["it ", "this ", "these ", "those ", "they ", "he ", "she "]
        if any(value.lower().startswith(start) for start in forbidden_starts):
            raise ValueError("Text starts with an ambiguous pronoun. Please use the Subject (Brand Name/Product) explicitly.")

        # Rule 3: Subject-first requirement (avoid single-word pronouns even when capitalized)
        first_word = value.strip().split()[0].strip(",.?!:;\"'\(\)").lower()
        if first_word in {"it", "this", "these", "those", "they", "he", "she"}:
            raise ValueError("First word must be an explicit subject (product/brand), not a pronoun.")

        # Rule 4: Single-paragraph constraint
        if "\n\n" in value or "\n" in value:
            raise ValueError("Answer Block must be a single paragraph without line breaks.")

        # Rule 5: Causal connector requirement
        if not any(connector in value.lower() for connector in {" because ", " therefore", " which means"}):
            raise ValueError("Answer Block must include a causal explanation using 'because', 'therefore', or 'which means'.")

        return value

    @field_validator("heading")
    def heading_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("heading cannot be blank")
        return value.strip()

    @field_validator("relevance_score")
    def score_range(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("relevance_score must be between 0-100")
        return value


class FanOutResult(BaseModel):
    main_keyword: str
    analysis_summary: str = Field(
        ..., 
        description="A brief strategic summary of why these fan-out queries were selected."
    )
    blocks: List[AnswerBlock]

    @model_validator(mode="after")
    def validate_blocks(self) -> "FanOutResult":
        block_count = len(self.blocks)
        if not 3 <= block_count <= 5:
            raise ValueError(f"Expected between 3-5 Answer Blocks, received {block_count}")

        queries = [block.target_query.lower() for block in self.blocks]
        if len(queries) != len(set(queries)):
            raise ValueError("target_query values must be unique")

        return self


# --- 2. AI Engine (The Architect) ---

def _build_system_prompt() -> str:
    return textwrap.dedent(
        """
        You are a top-tier GEO (Generative Engine Optimization) expert.
        Your task is to analyze the provided blog content and detect 'Fan-Out' queries.

        GEO RULES:
        1. AI models (ChatGPT, Google AI) only read 'Answer Blocks'.
        2. Each block must be able to stand alone (Standalone).
        3. Never write 'Intro' or 'Conclusion' sentences. Provide the direct answer.
        4. If numerical data (Price, Limit, Percentage) is available, you must use it.
        5. Try to use the 'Because / Therefore' logical structure.

        LMP (Language Model Pipeline) SIMULATION:
        - Score every block you generate between 0-100. If the text does not fully answer the question, lower the score.

        FAN-OUT RESEARCH PRINCIPLES:
        - Fan-out queries should cluster under the same SERP intent; avoid repetitive integration/feature highlights.
        - Prioritize high-volume variations ("price", "integration", "security", "alternatives").
        - Each block must answer a single problem or decision point.
        """
    ).strip()


def _build_user_prompt(content_text: str, keyword: str) -> str:
    trimmed_content = f"{content_text[:4000]}... (Content truncated)" if len(content_text) > 4000 else content_text
    return textwrap.dedent(
        f"""
        Content to Analyze:
        ---
        {trimmed_content}
        ---

        Target Keyword: "{keyword}"

        Find 3-5 sub-queries (Fan-Out) related to this keyword that users might ask but are not found as clear 'Snippets' in the text.
        Create Answer Blocks for each one that strictly follow GEO rules. Repeat the main query in every heading and keep the snippet focused in a single paragraph.
        """
    ).strip()


def generate_geo_content(content_text: str, keyword: str) -> Optional[FanOutResult]:
    """
    Analyzes content, finds fan-out queries, and generates validated blocks.
    
    Returns None if the model call fails.
    """

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(content_text, keyword)

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",  # Model supporting Structured Outputs
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=FanOutResult,
        )

        return completion.choices[0].message.parsed

    except Exception as exc:  # noqa: BLE001 - broad catch intentionally
        logger.exception("An error occurred during GEO content generation: %s", exc)
        return None


# --- 3. Test Run ---

def run_tool(content_text: str, keyword: str) -> dict:
    """
    Execute the FanOut AI workflow and return structured results for UI rendering.

    Parameters
    ----------
    content_text:
        The blog or article text to analyze.
    keyword:
        The primary keyword that guides fan-out query generation.

    Returns
    -------
    dict
        A dictionary containing either a ``result`` key with ``FanOutResult`` data
        or an ``error`` key with a human-friendly message.
    """

    content_text = (content_text or "").strip()
    keyword = (keyword or "").strip()

    if not content_text:
        return {"error": "Content is required via text input or uploaded file."}
    if not keyword:
        return {"error": "Keyword is required."}

    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY is not configured."}

    result = generate_geo_content(content_text, keyword)

    if result is None:
        return {"error": "The AI service could not generate a response. Please try again."}

    return {"result": result}
