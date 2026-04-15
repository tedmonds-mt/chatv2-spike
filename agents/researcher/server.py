from strands import Agent
from strands.multiagent.a2a.executor import StrandsA2AExecutor
from bedrock_agentcore.runtime import serve_a2a
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

researcher_agent = Agent(
    name="ResearcherAgent",
    description="A specialised research agent that provides factual summaries.",
    system_prompt="You are an expert researcher. Provide detailed, factual information on the requested topic.",
    model="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    trace_attributes={"service.name": "ResearcherAgent", "protocol": "A2A"},
)

if __name__ == "__main__":
    serve_a2a(StrandsA2AExecutor(researcher_agent), host="0.0.0.0", port=9000)
