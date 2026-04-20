import logging
import os
import sys

import boto3
from bedrock_agentcore.runtime import serve_a2a
from strands import Agent
from strands.multiagent.a2a.executor import StrandsA2AExecutor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

RESEARCHER_RUNTIME_ARN = os.environ.get("RESEARCHER_RUNTIME_ARN")
PROMPT_ARN = "arn:aws:bedrock:eu-west-2:281868401169:prompt/YBB8Q6KWHP:1"
REGION = os.environ.get("AWS_REGION", "eu-west-2")


def get_managed_prompt() -> str:
    """Retrieves the central prompt from Bedrock Prompt Management."""
    if not PROMPT_ARN:
        return "You are a technical orchestrator. Use the 'ask_researcher' tool to gather facts."
    bedrock_client = boto3.client("bedrock-agent", region_name=REGION)
    suffix = (
        "\n\nCRITICAL INSTRUCTION: If the 'ask_researcher' tool returns a CRITICAL SYSTEM ERROR, "
        "you must stop immediately and output the exact error text to the user. Do not write the "
        "article."
    )
    return (
        bedrock_client.get_prompt(promptIdentifier=PROMPT_ARN)["variants"][0][
            "templateConfiguration"
        ]["text"]["text"]
        + suffix
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
