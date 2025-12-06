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
logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. Data Models ---

class SubQueryPlan(BaseModel):
    main_intent: str
    sub_queries: List[str]


class AnswerBlock(BaseModel):
    heading: str
    content: str
    intent_category: Literal["Definition", "Comparison", "Limitations", "How-to"]
    source_quality_score: int
    relevance_score: int


class FanOutResult(BaseModel):
    analysis_summary: str
    blocks: List[AnswerBlock]


# --- 2. Prompt Builders ---

def _build_system_prompt() -> str:
    return textwrap.dedent("""
You are a GEO Content Engine specialized in AI-extractable answers.

Use the provided SEARCH CONTEXT as background knowledge, not as text to copy.
Each AnswerBlock must function as a standalone answer that can be surfaced
directly by AI systems without surrounding page context.

RULES:
- Generate between 3 and 5 AnswerBlocks in total.
- Create exactly one AnswerBlock per sub-query provided in the User Prompt.
- If fewer than 3 sub-queries are provided, synthesize additional high-value
  sub-queries yourself (based on the same keyword and intent) so that you still
  output at least 3 AnswerBlocks.
- Evaluate the quality of the provided Source text for each sub-query.

SOURCE USAGE LOGIC:
- If the Source text clearly and completely answers the query:
  - You may use the core idea from the Source
  - You MUST rewrite it significantly for clarity, authority, and GEO optimization
  - Set source_quality_score in the 70–90 range for strong sources

- If the Source text is weak, incomplete, or scattered:
  - Generate a better answer using your own subject-matter knowledge
  - Treat the Source only as contextual support
  - Set source_quality_score below 70 and enrich the answer accordingly

CONTENT REQUIREMENTS:
- AnswerBlock.content must be 40–80 words in a single paragraph.
- Start with an explicit subject (concept, product, or practice), never a pronoun.
- Deliver the core answer within the first one or two sentences (snippet-first).
- Include a clear causal explanation using "because", "therefore", or "which means".
- Close the decision or question; avoid vague or exploratory language.

SCORING GUIDANCE:
- source_quality_score reflects the usefulness of the original Source,
  not the quality of the generated AnswerBlock.
- For well-formed, GEO-compliant answers with reasonably good sources,
  prefer scores in the 70–85 range instead of being overly conservative.
- Only use scores below 50 when the Source is clearly weak or irrelevant.
- Prioritize GEO suitability and AI extractability over encyclopedic completeness.
""")


def _build_user_prompt(keyword: str, plan: SubQueryPlan, context: str) -> str:
    return textwrap.dedent(f"""
    Main Keyword: {keyword}
    Strategy: {plan.main_intent}

    SEARCH CONTEXT:
    {context}

    Generate a FanOutResult JSON.
    """)


# --- 3. Search Tool (Mock Only) ---

class SearchTool:
    def search_multiple(self, queries: List[str]) -> Dict[str, str]:
        results = {}
        for q in queries:
            results[q] = self._mock_search(q)
        return results

    def _mock_search(self, query: str) -> str:
        logger.info("[Smart Mock] Generating synthetic content for: %s", query)
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are simulating a Google search snippet. "
                        "Provide a concise, factual 100-word answer."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        return completion.choices[0].message.content


# --- 4. STEP 1 — GEO FAN-OUT DISCOVERY (UPDATED ✅) ---

def step_1_plan_queries(keyword: str, content_text: str = "") -> SubQueryPlan:
    system_prompt = textwrap.dedent("""
    You are a GEO Query Strategist.

    Generate fan-out queries optimized for AI answer surfaces.

    Use these intent templates:
    - Definition / Meaning
    - Importance / Why it matters
    - How-to / Process
    - Rules / Requirements
    - Consequences / Edge cases

    RULES:
    - Generate 3–5 distinct questions.
    - One question per intent.
    - Avoid paraphrases.
    - Must be answerable in one authoritative paragraph.
    """)

    user_prompt = textwrap.dedent(f"""
    Target Keyword: {keyword}

    Optional Content Context:
    {content_text[:1200] if content_text else "No content provided."}

    Output a SubQueryPlan JSON.
    """)

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=SubQueryPlan,
    )

    return completion.choices[0].message.parsed


# --- 5. STEP 3 — SYNTHESIS ---

def step_3_synthesize(
    keyword: str,
    plan: SubQueryPlan,
    search_context: Dict[str, str],
) -> FanOutResult:

    context_str = "\n\n".join(
        f"### Source for '{q}':\n{txt}" for q, txt in search_context.items()
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {
                "role": "user",
                "content": _build_user_prompt(keyword, plan, context_str),
            },
        ],
        response_format=FanOutResult,
    )

    return completion.choices[0].message.parsed


# --- 6. MAIN WORKFLOW ---

def run_fan_out_workflow(keyword: str, content_text: str = "") -> dict:
    try:
        plan = step_1_plan_queries(keyword, content_text)
        search_tool = SearchTool()
        search_results = search_tool.search_multiple(plan.sub_queries)
        final_result = step_3_synthesize(keyword, plan, search_results)
        return {"result": final_result.model_dump()}
    except Exception as exc:
        logger.exception("Fan-out workflow failed")
        return {"error": str(exc)}


# --- 7. Flask Adapter ---

def run_tool(content_text: str, keyword: str) -> dict:
    if not keyword:
        return {"error": "Keyword is required."}

    return run_fan_out_workflow(keyword, content_text)
