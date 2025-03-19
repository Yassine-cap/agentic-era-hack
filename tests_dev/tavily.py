from langchain_community.tools import TavilySearchResults
import getpass
import os
import json
import datetime
from rich.pretty import pprint

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
    
    
tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

test = tool.invoke({"query": "What are the job trends in 2025 ?"})
pprint(test)

for result in test:
    result["date"] = datetime.datetime.now().isoformat()
    result["question"] = "What are the job trends in 2025 ?"

path = "data_tavily"
os.makedirs(path, exist_ok=True)
with open(f"{path}/tavily_search.json", "w") as f:
    json.dump(test, f, indent=4)
    
pprint(f"Results saved in {path}/tavily_search.json")