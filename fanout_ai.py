import logging
import os
import json
import textwrap
from typing import List, Dict, Optional, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

# --- Configuration ---
load_dotenv()
logger = logging.getLogger(__name__)

# You would typically set this in your .env file
# OPENAI_API_KEY=sk-...
# SERPER_API_KEY=... (Optional: for real search)

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

# --- 2. The Search Tool (Retrieval Layer) ---

class SearchTool:
    """
    Handles the 'Retrieval' step in your diagram.
    Currently defaults to a MOCK implementation to save you money/setup time.
    """
    
    def __init__(self, provider: Literal["mock", "serper"] = "mock"):
        self.provider = provider
        self.api_key = os.getenv("SERPER_API_KEY")

    def search_multiple(self, queries: List[str]) -> Dict[str, str]:
        """
        Takes a list of queries, performs searches, and returns a 
        dictionary mapping 'query' -> 'summarized_search_results'.
        """
        results = {}
        for q in queries:
            if self.provider == "serper" and self.api_key:
                results[q] = self._search_serper(q)
            else:
                results[q] = self._search_mock(q)
        return results

    def _search_mock(self, query: str) -> str:
        """Simulates a search engine result for testing."""
        logger.info(f"[Mock Search] Searching for: {query}")
        # Return fake "scraped" content relevant to the query
        return f"[Simulated Search Result for '{query}']: Top result discusses {query} in detail. Key specs include 10 hour battery life and M2 processor benchmarks..."

    def _search_serper(self, query: str) -> str:
        """Real implementation using Serper.dev (Cheap/Fast)."""
        import requests
        
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}
        
        try:
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            # Extract snippets to form a context string
            snippets = [
                f"- {item.get('title')}: {item.get('snippet')}" 
                for item in data.get("organic", [])[:4] # Top 4 results
            ]
            return "\n".join(snippets)
        except Exception as e:
            logger.error(f"Search failed for {query}: {e}")
            return "No results found due to error."

# --- 3. The AI Chain (Architect) ---

def step_1_plan_queries(keyword: str) -> SubQueryPlan:
    """
    Step 1: Analyze the request and generate fan-out queries.
    """
    system_prompt = textwrap.dedent("""
        You are a Search Strategist. 
        Your goal is to break down a broad keyword into 3-5 specific sub-questions 
        that cover different aspects (Price, Specs, Reviews, Alternatives).
        These queries will be fed into a search engine.
    """)
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Target Keyword: {keyword}"}
        ],
        response_format=SubQueryPlan
    )
    return completion.choices[0].message.parsed

def step_3_synthesize(keyword: str, plan: SubQueryPlan, search_context: Dict[str, str]) -> FanOutResult:
    """
    Step 3: Combine the Search Context into final Answer Blocks.
    """
    # Format the search context for the AI
    context_str = ""
    for query, result_text in search_context.items():
        context_str += f"\n### Source for '{query}':\n{result_text}\n"

    system_prompt = textwrap.dedent("""
        You are a GEO Content Engine.
        Use the provided SEARCH CONTEXT to answer the questions.
        
        RULES:
        - Create exactly one AnswerBlock per sub-query provided in the User Prompt.
        - Use the specific 'Source' text provided for that query to write the answer.
        - If the source text is weak, use your own knowledge and mark source_quality_score low.
        - AnswerBlock.content must be 40-80 words, single paragraph.
    """)

    user_prompt = f"""
    Main Keyword: {keyword}
    Strategy Summary: {plan.main_intent}
    
    Background Data (Retrieved from Search):
    {context_str}
    
    Task: Generate the FanOutResult.
    """

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=FanOutResult
    )
    return completion.choices[0].message.parsed

# --- 4. Main Workflow ---

def run_fan_out_workflow(keyword: str, use_real_search: bool = False) -> dict:
    """
    Executes the full Pipeline: Plan -> Retrieve -> Synthesize
    """
    print(f"--- Starting Fan-Out for '{keyword}' ---")
    
    # 1. PLAN
    try:
        plan = step_1_plan_queries(keyword)
        print(f"✅ Step 1 (Plan): Generated {len(plan.sub_queries)} queries.")
        for q in plan.sub_queries:
            print(f"   - {q}")
    except Exception as e:
        return {"error": f"Planning failed: {str(e)}"}

    # 2. RETRIEVE
    try:
        provider = "serper" if use_real_search else "mock"
        search_tool = SearchTool(provider=provider)
        search_results = search_tool.search_multiple(plan.sub_queries)
        print(f"✅ Step 2 (Retrieval): Fetched data for {len(search_results)} queries using {provider}.")
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

    # 3. SYNTHESIZE
    try:
        final_result = step_3_synthesize(keyword, plan, search_results)
        print(f"✅ Step 3 (Synthesis): Generated {len(final_result.blocks)} answer blocks.")
    except Exception as e:
        return {"error": f"Synthesis failed: {str(e)}"}

    return final_result.model_dump()

# --- Example Usage ---
if __name__ == "__main__":
    # Test run
    result = run_fan_out_workflow("Best laptop for college students")
    print(json.dumps(result, indent=2))
