import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
# Data visualization libraries
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
# Use ChatVertexAI instead of ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
# LangGraph imports
from langgraph.graph import END, START, StateGraph

load_dotenv()


# Set your API keys and project id
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")  # Replace with your Google Cloud project ID
MODEL_ID = "gemini-2.0-flash-001"
THINKING_MODEL_ID = "gemini-2.0-flash-thinking-exp-01-21"
# Configure logging
from loguru import logger

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

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

# Initialize the LLM using ChatVertexAI
llm = ChatVertexAI(
    project=PROJECT_ID,
    model=MODEL_ID,  # Use the appropriate model name in your project
    temperature=0.2
)

think_llm = ChatVertexAI(
    project=PROJECT_ID,
    model=MODEL_ID,  # Use the appropriate model name in your project
    temperature=0.3
)

# Initialize Tavily Search Tool
tavily_search = TavilySearchResults(
    max_results=10,
    include_domains=["linkedin.com", "indeed.com", "glassdoor.com", "weforum.org", "bls.gov"],
    exclude_domains=["pinterest.com", "facebook.com"]
)

def merge_state(state: WorkforceState, updates: Dict[str, Any]) -> WorkforceState:
    """
    Helper function to merge new updates into the existing state.
    """
    state.update(updates)
    return state

# ----------------------------
# Step 1: Collect Search Data
# ----------------------------
def collect_search_data(state: WorkforceState) -> WorkforceState:
    """
    Perform a search for global workforce trends based on the query in the state.
    """
    if state.get("error"):
        return state  # Skip processing if an error exists

    query = state["query"]
    search_query = f"latest global workforce trends for {query} skills demand statistics"
    
    try:
        search_results = tavily_search.invoke(search_query)
        logger.info("Search completed successfully.")
        return merge_state(state, {"search_results": search_results, "status": "search_completed"})
    except Exception as e:
        logger.error("Search failed: %s", str(e))
        return merge_state(state, {"error": str(e), "status": "search_failed"})

# ----------------------------
# Step 2: Analyze Trends
# ----------------------------
def analyze_trends(state: WorkforceState) -> WorkforceState:
    """
    Analyze search results to identify top in-demand skills.
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
    truncated_content = content_combined[:8000]  # Limit content length
    
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

# ----------------------------
# Step 3: Aggregate Data
# ----------------------------
def aggregate_data(state: WorkforceState) -> WorkforceState:
    """
    Aggregate trend data by categorizing skills and computing statistics.
    """
    if state.get("error"):
        return state

    trends = state.get("trends", [])
    if not trends:
        error_msg = "No trends data available to aggregate."
        logger.error(error_msg)
        return merge_state(state, {"error": error_msg, "status": "aggregation_failed"})
    
    # Group skills by categories
    categories = {}
    for skill in trends:
        category = skill.get("category", "unknown")
        categories.setdefault(category, []).append(skill)
    
    aggregated_data = {
        "categories": {},
        "overall_stats": {
            "total_skills": len(trends),
            "avg_demand": sum(s.get("demand_level", 0) for s in trends) / len(trends),
            "avg_growth": sum(s.get("growth_rate", 0) for s in trends) / len(trends)
        }
    }
    
    for category, skills in categories.items():
        avg_demand = sum(s.get("demand_level", 0) for s in skills) / len(skills)
        avg_growth = sum(s.get("growth_rate", 0) for s in skills) / len(skills)
        
        aggregated_data["categories"][category] = {
            "skills": sorted(skills, key=lambda x: x.get("demand_level", 0), reverse=True),
            "avg_demand": avg_demand,
            "avg_growth": avg_growth,
            "count": len(skills)
        }
    
    logger.info("Data aggregation completed successfully.")
    return merge_state(state, {"aggregated_data": aggregated_data, "status": "data_aggregated"})

# ----------------------------
# Step 4: Create Data Visualizations
# ----------------------------
def create_visualizations(state: WorkforceState) -> WorkforceState:
    """
    Create visualizations based on trends and aggregated data.
    """
    if state.get("error"):
        return state

    aggregated_data = state.get("aggregated_data", {})
    trends = state.get("trends", [])
    if not trends:
        error_msg = "No trends data available for visualization."
        logger.error(error_msg)
        return merge_state(state, {"error": error_msg, "status": "visualization_failed"})
    
    visualizations = []
    
    # Create DataFrame from trends data
    df = pd.DataFrame(trends)
    
    # 1. Bar chart of top skills by demand
    top_skills = df.sort_values(by="demand_level", ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x="demand_level", y="skill_name", data=top_skills, palette="viridis")
    plt.title("Top 10 Skills by Demand Level")
    plt.tight_layout()
    
    buf1 = BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    img_base64_1 = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close()
    
    visualizations.append({
        "type": "image",
        "title": "Top 10 Skills by Demand Level",
        "data": img_base64_1
    })
    
    # 2. Scatter plot of demand vs growth
    # Function to determine offset dynamically
    def get_offset(demand, growth):
        """Adjust text position dynamically based on quadrant"""
        offset_x = 0.2 if demand < 8 else -0.2  # Move right if demand is low, left otherwise
        offset_y = 0.2 if growth < 2 else -0.2  # Move up if growth is low, down otherwise
        return offset_x, offset_y
    
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="demand_level", y="growth_rate", hue="category", size="demand_level", 
                    sizes=(50, 200), alpha=0.7, data=df)
    
    for i, row in df.iterrows():
        offset_x, offset_y = get_offset(row['demand_level'], row['growth_rate'])
        # Apply a small jitter to avoid perfect overlaps
        jitter_x = np.random.uniform(-0.1, 0.1)
        jitter_y = np.random.uniform(-0.1, 0.1)
        plt.text(
            row['demand_level'] + jitter_x + offset_x, 
            row['growth_rate'] + jitter_y + offset_y, 
            row['skill_name'], 
            fontsize=9, fontweight='bold',
            ha='left' if offset_x > 0 else 'right',  # Adjust horizontal alignment
            va='bottom' if offset_y > 0 else 'top'  # Adjust vertical alignment
        )
        # plt.text(row['demand_level']+ 0.2, row['growth_rate']+ 0.2, row['skill_name'], fontsize=12, fontweight='bold', ha='left', va='bottom')
    
    plt.title("Skills Positioning: Demand vs Growth", fontsize=14, fontweight='bold')
    plt.xlabel("Current Demand Level (1-10)", fontsize=12)
    plt.ylabel("Growth Rate (-5 to +5)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Category", fontsize=10)
    plt.tight_layout()
    
    buf2 = BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    img_base64_2 = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close()
    
    visualizations.append({
        "type": "image",
        "title": "Skills Positioning: Demand vs Growth",
        "data": img_base64_2
    })
    
    # 3. Heatmap of categories
    categories = aggregated_data.get("categories", {})
    cat_data = {
        cat: {
            "avg_demand": data.get("avg_demand", 0), 
            "avg_growth": data.get("avg_growth", 0), 
            "count": data.get("count", 0)
        } for cat, data in categories.items()
    }
    
    cat_df = pd.DataFrame(cat_data).T
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cat_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Category Analysis: Demand, Growth and Skill Count")
    plt.tight_layout()
    
    buf3 = BytesIO()
    plt.savefig(buf3, format='png')
    buf3.seek(0)
    img_base64_3 = base64.b64encode(buf3.read()).decode('utf-8')
    plt.close()
    
    visualizations.append({
        "type": "image",
        "title": "Category Analysis",
        "data": img_base64_3
    })
    
    logger.info("Visualizations created successfully.")
    return merge_state(state, {"visualizations": visualizations, "status": "visualizations_created"})

# ----------------------------
# Step 5: Generate Report
# ----------------------------
def generate_report(state: WorkforceState) -> WorkforceState:
    """
    Generate a comprehensive HR workforce trends report.
    """
    if state.get("error"):
        return state

    query = state["query"]
    trends = state.get("trends", [])
    aggregated_data = state.get("aggregated_data", {})
    trends_data = json.dumps(trends, indent=2)
    aggregated_data = json.dumps(aggregated_data, indent=2)
    
    report_prompt = """\
    Your goal is to offer actionable insights to help businesses transition smoothly with the GenAI revolution in the workforce.

    Your task is to generate a **Personalized Reskilling Recommendation Report** related to "{query}" trends and workforce gaps.

    Your report must be based on these trends and aggregated data:

    TRENDS DATA:
    {trends_data}

    AGGREGATED DATA:
    {aggregated_data}

    The report must follow this structure:

    {{
        "executive_summary": "A concise summary of the report's purpose and key takeaways.",
        "industry_insights": {{
            "key_trends": [
                {{
                    "title": "Trend Title",
                    "description": "Brief explanation of the trend's impact on the workforce."
                }}
            ],
            "challenges": [
                {{
                    "title": "Challenge Title",
                    "description": "Description of the business challenge related to AI and data."
                }}
            ]
        }},
        "skills_demand": {{
            "technical_skills": [
                {{
                    "name": "Skill Name",
                    "demand_level": 0-10,
                    "growth_rate": 0-5
                }}
            ],
            "cognitive_skills": [
                {{
                    "name": "Skill Name",
                    "demand_level": 0-10,
                    "growth_rate": 0-5
                }}
            ],
            "soft_skills": [
                {{
                    "name": "Skill Name",
                    "demand_level": 0-10,
                    "growth_rate": 0-5
                }}
            ]
        }},
        "workforce_gaps": {{
            "technical": "Analysis of gaps in technical skills",
            "cognitive": "Analysis of gaps in cognitive skills",
            "soft_skills": "Analysis of gaps in soft skills"
        }},
        "reskilling_recommendations": {{
            "technical_skills": [
                {{
                    "name": "Skill Name",
                    "training_programs": ["Program 1", "Program 2"],
                    "certifications": ["Certification 1", "Certification 2"],
                    "learning_paths": ["Learning Path 1", "Learning Path 2"]
                }}
            ],
            "cognitive_skills": [
                {{
                    "name": "Skill Name",
                    "training_programs": ["Program 1", "Program 2"],
                    "learning_paths": ["Learning Path 1", "Learning Path 2"]
                }}
            ],
            "soft_skills": [
                {{
                    "name": "Skill Name",
                    "training_programs": ["Program 1", "Program 2"],
                    "learning_paths": ["Learning Path 1", "Learning Path 2"]
                }}
            ]
        }},
        "conclusion": "A summary of the key takeaways and next steps for businesses."
    }}

    ### Output Instructions:
    - The response **must** be a valid JSON object.
    - Ensure all elements are well-structured, correctly nested, and formatted for readability.
    - Use **arrays** where multiple elements exist, such as skills or challenges.
    - Provide **concise descriptions** for each section.
    - Use **consistent field names** as specified above.
    - **DO NOT** generate text output in paragraph format; output must strictly follow the JSON schema.
    """

    report_prompt = report_prompt.format(query=query, trends_data=trends_data, aggregated_data=aggregated_data)

    
    try:
        response = think_llm.invoke([HumanMessage(content=report_prompt)])
        report = response.content
        logger.info("Report generated successfully.")
        return merge_state(state, {"report": report, "status": "report_generated"})
    except Exception as e:
        logger.error("Report generation failed: %s", str(e))
        return merge_state(state, {"error": f"Error generating report: {str(e)}", "status": "report_generation_failed"})

# ----------------------------
# Build the Workflow Graph
# ----------------------------
workflow = StateGraph(WorkforceState)

# Add nodes to the workflow

workflow.add_node("search_data", collect_search_data)
workflow.add_node("analyze_trends", analyze_trends)
workflow.add_node("aggregate_data", aggregate_data)
workflow.add_node("create_visualizations", create_visualizations)
workflow.add_node("generate_report", generate_report)

# Define the execution edges
workflow.add_edge(START, "search_data")
workflow.add_edge("search_data", "analyze_trends")
workflow.add_edge("analyze_trends", "aggregate_data")
workflow.add_edge("aggregate_data", "create_visualizations")
workflow.add_edge("create_visualizations", "generate_report")
workflow.add_edge("generate_report", END)

# Add conditional error handling edges
def handle_errors(state: WorkforceState) -> str:
    return "handle_error" if state.get("error") else "continue"

workflow.add_conditional_edges(
    "search_data",
    handle_errors,
    {"handle_error": END, "continue": "analyze_trends"}
)

workflow.add_conditional_edges(
    "analyze_trends",
    handle_errors,
    {"handle_error": END, "continue": "aggregate_data"}
)

# Compile the workflow into an application instance
app = workflow.compile()

# ----------------------------
# Main Testing Block
# ----------------------------
# if __name__ == "__main__":
#     # Define an initial state with a sample query (modify the query as needed)
#     initial_state: WorkforceState = {
#         "query": "data science",
#         "search_results": [],
#         "trends": [],
#         "aggregated_data": {},
#         "visualizations": [],
#         "report": "",
#         "status": "initialized",
#         "error": None
#     }
    
#     # Run the workflow
#     final_state = app.invoke(initial_state)
    
#     # Print the final state in a readable JSON format
#     print("Final Workflow State:")
#     print(json.dumps(final_state, indent=2))
