import logging
import os
import json
import textwrap
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# --- Configuration ---
load_dotenv()
logger = logging.getLogger(__name__)

# Load API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- 1. Data Models ---

class SubQueryPlan(BaseModel):
    """Step 1 Output: The strategic plan."""

    main_intent: str = Field(description="The primary intent of the user (e.g., 'Commercial Investigation').")
    sub_queries: List[str] = Field(description="List of 3-5 specific search queries to gather missing information.")


class AnswerBlock(BaseModel):
    """Step 3 Output: The final content block."""

    heading: str
    content: str = Field(description="The answer text (40-80 words).")
    intent_category: Literal["Definition", "Comparison", "Limitations", "How-to"]
    source_quality_score: int
    relevance_score: int


class FanOutResult(BaseModel):
    """The final result object."""

    analysis_summary: str
    blocks: List[AnswerBlock]


# --- 2. Prompt Builders ---

def _build_system_prompt() -> str:
    """Return the system prompt used when generating GEO content directly."""

    return textwrap.dedent(
        """
        You are a GEO Content Engine.
        Produce concise, well-structured answer blocks based on the provided content and keyword.

        Rules:
        - Create one AnswerBlock per synthesized subtopic.
        - Favor clarity and factual tone.
        - Keep AnswerBlock.content between 40-80 words.
        - Score source_quality_score higher when the provided background looks reliable; otherwise keep it low.
        - Set relevance_score based on how tightly the block answers the keyword intent.
        """
    )


def _build_user_prompt(content_text: str, keyword: str) -> str:
    """Shape the user prompt for the GEO content generator."""

    base_content = content_text.strip() or "No additional background text provided."
    return textwrap.dedent(
        f"""
        Main Keyword: {keyword}
        Background Content:
        {base_content}

        Please analyze the keyword and background to produce a FanOutResult JSON.
        """
    )


# --- 3. The Search Tool (Retrieval Layer) ---

class SearchTool:
    """
    Handles the 'Retrieval' step.
    Default: 'mock' mode uses GPT-4o to SIMULATE search results (Synthetic Data).
    """

    def __init__(self, provider: Literal["mock", "serper"] = "mock"):
        self.provider = provider
        self.api_key = os.getenv("SERPER_API_KEY")

    def search_multiple(self, queries: List[str]) -> Dict[str, str]:
        results = {}
        for q in queries:
            if self.provider == "serper" and self.api_key:
                results[q] = self._search_serper(q)
            else:
                results[q] = self._search_mock_smart(q)
        return results

    def _search_mock_smart(self, query: str) -> str:
        """
        Uses the LLM's internal knowledge to hallucinate a realistic search snippet.
        This makes the tool functional for testing/drafting without a real search engine.
        """

        logger.info("[Smart Mock] Generating synthetic content for: %s", query)

        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a search engine simulator. Write a realistic, high-quality "
                            "search result snippet (100 words) that directly answers the user's query with facts."
                        ),
                    },
                    {"role": "user", "content": f"Search Query: {query}"},
                ],
            )
            return completion.choices[0].message.content
        except Exception as exc:  # pragma: no cover - network/LLM dependent
            logger.error("Smart mock failed: %s", exc)
            return "Error generating mock content."

    def _search_serper(self, query: str) -> str:
        """Real implementation using Serper.dev"""

        import requests

        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        try:
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            snippets = [
                f"- {item.get('title')}: {item.get('snippet')}"
                for item in data.get("organic", [])[:4]
            ]
            return "\n".join(snippets)
        except Exception as exc:  # pragma: no cover - network/LLM dependent
            logger.error("Search failed for %s: %s", query, exc)
            return "No results found due to error."


# --- 4. The AI Chain (Architect) ---

def step_1_plan_queries(keyword: str) -> SubQueryPlan:
    """Step 1: Analyze the request and generate fan-out queries."""

    system_prompt = textwrap.dedent(
        """
        You are a Search Strategist.
        Your goal is to break down a broad keyword into 3-5 specific sub-questions
        that cover different aspects (Price, Specs, Reviews, Alternatives).
        These queries will be fed into a search engine.
        """
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Target Keyword: {keyword}"},
        ],
        response_format=SubQueryPlan,
    )
    return completion.choices[0].message.parsed


def step_3_synthesize(keyword: str, plan: SubQueryPlan, search_context: Dict[str, str]) -> FanOutResult:
    """Step 3: Combine the Search Context into final Answer Blocks."""

    context_parts = [f"### Source for '{query}':\n{result_text}" for query, result_text in search_context.items()]
    context_str = "\n\n".join(context_parts)

    system_prompt = textwrap.dedent(
        """
        You are a GEO Content Engine.
        Use the provided SEARCH CONTEXT to answer the questions.

        RULES:
        - Create exactly one AnswerBlock per sub-query provided in the User Prompt.
        - Use the specific 'Source' text provided for that query to write the answer.
        - If the source text is weak, use your own knowledge and mark source_quality_score low.
        - AnswerBlock.content must be 40-80 words, single paragraph.
        """
    )

    user_prompt = textwrap.dedent(
        f"""
        Main Keyword: {keyword}
        Strategy Summary: {plan.main_intent}

        Background Data (Retrieved from Search):
        {context_str}

        Task: Generate the FanOutResult.
        """
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=FanOutResult,
    )
    return completion.choices[0].message.parsed


# --- 5. Main Workflow & Public APIs ---

def run_fan_out_workflow(keyword: str, use_real_search: bool = False) -> dict:
    """Executes the full Pipeline: Plan -> Retrieve -> Synthesize."""

    try:
        plan = step_1_plan_queries(keyword)
    except Exception as exc:  # pragma: no cover - network/LLM dependent
        logger.exception("Step 1 failed")
        return {"error": f"Planning failed: {str(exc)}"}

    try:
        provider = "serper" if use_real_search else "mock"
        search_tool = SearchTool(provider=provider)
        search_results = search_tool.search_multiple(plan.sub_queries)
    except Exception as exc:  # pragma: no cover - network/LLM dependent
        logger.exception("Step 2 failed")
        return {"error": f"Search failed: {str(exc)}"}

    try:
        final_result = step_3_synthesize(keyword, plan, search_results)
    except Exception as exc:  # pragma: no cover - network/LLM dependent
        logger.exception("Step 3 failed")
        return {"error": f"Synthesis failed: {str(exc)}"}

    return {"result": final_result.model_dump()}


def generate_geo_content(content_text: str, keyword: str) -> Optional[FanOutResult]:
    """Generate GEO content directly from provided text without a retrieval step."""

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(content_text, keyword)

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=FanOutResult,
        )

        return completion.choices[0].message.parsed
    except Exception as exc:  # pragma: no cover - network/LLM dependent
        logger.exception("An error occurred during GEO content generation: %s", exc)
        return None


# --- Flask Adapter ---

def run_tool(content_text: str, keyword: str) -> dict:
    """
    Adapter function to make the new Fan-Out logic compatible with
    the existing Flask app call signature.
    """

    if not keyword:
        return {"error": "Keyword is required."}

    # If the user provided source content, try to generate directly; otherwise use the fan-out workflow.
    if content_text.strip():
        geo_result = generate_geo_content(content_text, keyword)
        if geo_result is not None:
            return {"result": geo_result.model_dump()}

    # Fallback to the retrieval-driven workflow.
    return run_fan_out_workflow(keyword, use_real_search=False)
