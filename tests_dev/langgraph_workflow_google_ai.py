from langchain_google_vertexai import ChatVertexAI

# Replace with your actual Google Cloud project ID
project_id = "qwiklabs-gcp-02-2a44d1630c0c"

# Initialize the Vertex AI Chat model
chat_agent = ChatVertexAI(
    project=project_id,
    temperature=0.7,
    max_output_tokens=256
)

# Define a simple graph class to simulate the agent workflow
class AgentGraph:
    def __init__(self):
        self.nodes = {}
        self.execution_order = []
    
    def add_node(self, name, func):
        self.nodes[name] = func
        self.execution_order.append(name)
    
    def run(self, question: str):
        intermediate_results = {}
        for name in self.execution_order:
            func = self.nodes[name]
            result = func(question)
            intermediate_results[name] = result
            print(f"Result from node '{name}': {result}")
        return intermediate_results

# Function that uses the chat agent to generate a response
def agent_response(question: str):
    return chat_agent.generate(question)

# Build the agent graph by adding the response node
graph = AgentGraph()
graph.add_node("agent_response", agent_response)

# Example question to test the agent workflow
if __name__ == "__main__":
    question = "What is the capital of France?"
    results = graph.run(question)
    print("\nFinal Results:")
    print(results)