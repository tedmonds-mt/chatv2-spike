import os
from strands import Agent, tool
import sys
import boto3
import json

RESEARCHER_RUNTIME_ID = os.environ.get("RESEARCHER_RUNTIME_ID")


@tool
def ask_researcher(topic: str) -> str:
    """Delegates deep research to the researcher agent via Bedrock AgentCore"""
    if not RESEARCHER_RUNTIME_ID:
        return "Error: Researcher runtime ID not configured."

    agent_core_client = boto3.client("bedrock-agentcore")

    payload = json.dumps({"prompt": f"Provide a detailed factual survey of {topic}"})

    response = agent_core_client.invoke_agent_runtime(
        agent_runtime_arn=RESEARCHER_RUNTIME_ID, payload=payload
    )

    response_body = response["payload"].read().decode("utf-8")
    return json.loads(response_body).get("result", response_body)


writer_agent = Agent(
    name="WriterAgent",
    system_prompt="You are a technical writer. Use the remote Researcher Agent to gather facts, then write an engaging paragraph.",
    tools=[ask_researcher],
)

if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "The key principles behind the A2A Protocol."
    )
    print(writer_agent(query))
