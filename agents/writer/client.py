import os
from strands import Agent, tool
from strands.multiagent.a2a.executor import StrandsA2AExecutor
from bedrock_agentcore.runtime import serve_a2a
import boto3
import json
import uuid

RESEARCHER_RUNTIME_ARN = os.environ.get("RESEARCHER_RUNTIME_ARN")


@tool
def ask_researcher(topic: str) -> str:
    """Delegates deep research to the researcher agent via Bedrock AgentCore"""
    if not RESEARCHER_RUNTIME_ARN:
        return "Error: Researcher runtime ARN not configured."

    client = boto3.client("bedrock-agentcore")

    a2a_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": f"Research {topic}"}],
            }
        },
    }

    encoded_payload = json.dumps(a2a_payload).encode("utf-8")

    response = client.invoke_agent_runtime(
        agent_runtime_arn=RESEARCHER_RUNTIME_ARN, payload=encoded_payload
    )

    response_body = json.loads(response["payload"].read().decode("utf-8"))

    try:
        return response_body["result"]["message"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return str(response_body)


writer_agent = Agent(
    name="WriterAgent",
    system_prompt=("You are a gonzo journalist. "
                   "Use the remote Researcher Agent to gather facts about the given topic, "
                   "then write an engaging paragraph in a sensational tabloid style summarising "
                   "those facts. "
                   "Make sure there is at least one spurious and outrageous exaggeration."
                   "Overuse CAPITAL LETTERS AND EXCLAMATIONS!"),

    tools=[ask_researcher],
)

if __name__ == "__main__":
    serve_a2a(StrandsA2AExecutor(writer_agent))
