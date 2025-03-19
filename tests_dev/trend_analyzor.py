import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state structure
class WorkforceState(TypedDict):
    query: str
    search_results: List[Dict]
    trends: List[Dict]
    aggregated_data: Dict
    visualizations: List[Dict]
    report: str
    status: str
    error: Optional[str]

# Set your Google Cloud project id
PROJECT_ID = "YOUR_PROJECT_ID"  # Replace with your actual project ID

# Initialize the LLM using ChatVertexAI
llm = ChatVertexAI(
    project=PROJECT_ID,
    model="chat-bison",  # Use the appropriate model name for your project
    temperature=0
)

def merge_state(state: WorkforceState, updates: Dict[str, Any]) -> WorkforceState:
    """
    Helper function to merge new updates into the existing state.
    """
    state.update(updates)
    return state

def analyze_trends(state: WorkforceState) -> WorkforceState:
    """
    Analyze search results to identify top in-demand skills.
    
    Expects state to have:
      - "query": a string representing the skill domain (e.g., "data science")
      - "search_results": a list of dictionaries each containing at least a "content" field
    Returns the updated state with a new "trends" key holding the JSON output.
    """
    if state.get("error"):
        return state

    search_results = state.get("search_results", [])
    query = state["query"]

    # Extract content from search results (if available)
    content_texts = [result["content"] for result in search_results if "content" in result]
    if not content_texts:
        error_msg = "No content available from search results."
        logger.error(error_msg)
        return merge_state(state, {"error": error_msg, "status": "trend_analysis_failed"})

    content_combined = "\n\n".join(content_texts)
    truncated_content = content_combined[:8000]  # Limit content length if needed

    trend_prompt = f"""
    Based on the following data sources about global workforce trends, identify the top 15 most in-demand skills related to {query}.
    
    DATA SOURCES:
    {truncated_content}
    
    Analyze this information and provide:
    1. The top 15 skills in JSON format
    2. For each skill include: 
       - "skill_name": The name of the skill
       - "demand_level": A numerical score from 1-10 indicating current demand
       - "growth_rate": A numerical score from -5 to 5 indicating growth trajectory
       - "category": The category this skill belongs to (e.g., technical, soft skill, domain knowledge)
    
    Format your response as a valid JSON list of objects, nothing else.
    """

    try:
        response = llm.invoke([HumanMessage(content=trend_prompt)])
        response_text = response.content

        # Extract JSON content using regex
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        json_str = json_match.group(0) if json_match else response_text

        trends = json.loads(json_str)
        logger.info("Trend analysis completed successfully.")
        return merge_state(state, {"trends": trends, "status": "trends_analyzed"})
    except Exception as e:
        logger.error("Trend analysis failed: %s", str(e))
        return merge_state(state, {"error": f"Error analyzing trends: {str(e)}", "status": "trend_analysis_failed"})


def main():
    # Create an initial state with a sample query and dummy search results
    initial_state: WorkforceState = {
        "query": "data science",
        "search_results": [
            {
                "content": (
                    "Data science is evolving rapidly with trends in machine learning, deep learning, "
                    "and data analytics. Technologies like Python, R, and SQL remain in high demand. "
                    "The market is also seeing growth in data visualization and cloud-based analytics solutions."
                )
            }
        ],
        "trends": [],
        "aggregated_data": {},
        "visualizations": [],
        "report": "",
        "status": "initialized",
        "error": None
    }

    # Invoke the analyze_trends agent
    final_state = analyze_trends(initial_state)

    # Print the final state in a readable JSON format for debugging
    print("Final state after analyze_trends:")
    print(json.dumps(final_state, indent=2))

if __name__ == "__main__":
    main()