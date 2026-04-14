import os
from strands.agent import Agent
from strands.agent.a2a_agent import A2AAgent
import sys

RESEARCHER_ENDPOINT = os.environ.get("RESEARCHER_ENDPOINT", "http://localhost:9000")

researcher = A2AAgent(endpoint=RESEARCHER_ENDPOINT)

writer_agent = Agent(
    name="WriterAgent",
    system_prompt="You are a technical writer. Use the remote Researcher Agent to gather facts, then write an engaging paragraph.",
    tools=[researcher]
)

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "The key principles behind the A2A Protocol."
    print(writer_agent(query))
