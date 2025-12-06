"""Fan-Out GEO content generator utilities.

This module exposes a pure `run_tool` function that wraps the original CLI
logic into a callable interface suitable for web applications or other Python
code. It handles validation, prompt construction, and OpenAI calls while
surfacing user-friendly exceptions for error handling.
"""

import os
import textwrap
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# Load environment variables early so consumers can rely on OPENAI_API_KEY.
load_dotenv()

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
        description="The likely user fan-out query (e.g., 'Jotform vs Zapier pricing')",
    )
    heading: str = Field(
        ...,
        description="The heading to be used as H2 or H3 in the blog post",
    )
    content: str = Field(
        ...,
        description="The snippet text containing the direct answer",
    )
    relevance_score: int = Field(
        ...,
        description="LMP Relevance Score (0-100). How relevant is the content to the question?",
    )

    @field_validator("intent_category")
    def validate_intent(cls, value: str) -> str:  # noqa: N805 - Pydantic validator signature
        if value not in _INTENT_CHOICES:
            raise ValueError(
                f"intent_category '{value}' is not supported. Valid options: {_INTENT_CHOICES}"
            )
        return value

    @field_validator("content")
    def validate_geo_rules(cls, value: str) -> str:  # noqa: N805 - Pydantic validator signature
        word_count = len(value.split())
        if word_count < 40 or word_count > 80:
            raise ValueError(
                f"Answer Block length does not meet GEO standards ({word_count} words). Must be between 40-80."
            )

        forbidden_starts = ["it ", "this ", "these ", "those ", "they ", "he ", "she "]
        if any(value.lower().startswith(start) for start in forbidden_starts):
            raise ValueError(
                "Text starts with an ambiguous pronoun. Please use the Subject (Brand Name/Product) explicitly."
            )

        return value

    @field_validator("heading")
    def heading_cannot_be_blank(cls, value: str) -> str:  # noqa: N805 - Pydantic validator signature
        if not value.strip():
            raise ValueError("heading cannot be blank")
        return value.strip()

    @field_validator("relevance_score")
    def score_range(cls, value: int) -> int:  # noqa: N805 - Pydantic validator signature
        if not 0 <= value <= 100:
            raise ValueError("relevance_score must be between 0-100")
        return value


class FanOutResult(BaseModel):
    main_keyword: str
    analysis_summary: str = Field(
        ...,
        description="A brief strategic summary of why these fan-out queries were selected.",
    )
    blocks: List[AnswerBlock]

    @model_validator(mode="after")
    def validate_blocks(self) -> "FanOutResult":  # noqa: N805 - Pydantic validator signature
        block_count = len(self.blocks)
        if not 3 <= block_count <= 5:
            raise ValueError(f"Expected between 3-5 Answer Blocks, received {block_count}")

        queries = [block.target_query.lower() for block in self.blocks]
        if len(queries) != len(set(queries)):
            raise ValueError("target_query values must be unique")

        return self


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Build a cached OpenAI client using the configured API key."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not configured. Set it in your environment or a .env file before running the tool."
        )

    return OpenAI(api_key=api_key)


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


def run_tool(content_text: str, keyword: str) -> FanOutResult:
    """Generate GEO-compliant fan-out content for the given text and keyword.

    Args:
        content_text: Source blog or article text to analyze.
        keyword: The main keyword to pivot fan-out research around.

    Returns:
        A validated ``FanOutResult`` containing the analysis summary and answer blocks.

    Raises:
        ValueError: If inputs are missing or invalid.
        EnvironmentError: If the API key is not configured.
        RuntimeError: If the OpenAI request fails or returns an unexpected payload.
    """

    if not content_text or not content_text.strip():
        raise ValueError("Content text is required to run the fan-out analysis.")
    if not keyword or not keyword.strip():
        raise ValueError("A target keyword is required to run the fan-out analysis.")

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(content_text.strip(), keyword.strip())

    client = _get_client()

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",  # Model supporting Structured Outputs
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=FanOutResult,
        )
    except Exception as exc:  # noqa: BLE001 - OpenAI SDK can raise various exceptions
        raise RuntimeError(f"Failed to generate GEO content: {exc}") from exc

    try:
        return completion.choices[0].message.parsed
    except (AttributeError, ValidationError, IndexError) as exc:
        raise RuntimeError("Received an unexpected response format from the model.") from exc


__all__ = [
    "AnswerBlock",
    "FanOutResult",
    "run_tool",
]
