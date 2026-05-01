import logging
import os
import sys

import boto3
import requests
from bedrock_agentcore.runtime import serve_a2a
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from strands import Agent
from strands.multiagent.a2a.executor import StrandsA2AExecutor
from strands.tools.mcp import MCPClient

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Environment variable '{key}' must be set")
    return value


CLIENT_ID = get_env("CLIENT_ID")
CLIENT_SECRET = get_env("CLIENT_SECRET")
GATEWAY_ID = get_env("TOKEN_URL")
MCP_URL = get_env("MCP_URL")
REGION = os.environ.get("AWS_REGION", "eu-west-2")

TOKEN_URL = f"https://{GATEWAY_ID}.auth.eu-west-2.amazoncognito.com/oauth2/token"

PROMPT_ARN = "arn:aws:bedrock:eu-west-2:715195480427:prompt/1NNKU1PDXX"


def get_managed_prompt() -> str:
    """Retrieves the central prompt from Bedrock Prompt Management."""
    if not PROMPT_ARN:
        return "You are a technical orchestrator. Use the 'ask_researcher' tool to gather facts."
    bedrock_client = boto3.client("bedrock-agent", region_name=REGION)

    return bedrock_client.get_prompt(promptIdentifier=PROMPT_ARN)["variants"][0][
        "templateConfiguration"
    ]["text"]["text"]


def fetch_access_token(client_id, client_secret, token_url):
    response = requests.post(
        token_url,
        data="grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}".format(
            client_id=client_id, client_secret=client_secret
        ),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return response.json()["access_token"]


def create_streamable_http_transport(mcp_url: str, access_token: str):
    client = create_mcp_http_client(headers={"Authorization": f"Bearer {access_token}"})
    return streamable_http_client(mcp_url, http_client=client)


mcp_access_token = fetch_access_token(CLIENT_ID, CLIENT_SECRET, TOKEN_URL)
streamable_http_mcp_client = MCPClient(
    lambda: create_streamable_http_transport(MCP_URL, mcp_access_token)
)


researcher_agent = Agent(
    name="ResearcherAgent",
    description="A specialised research agent to answer complex questions about the UK government.",
    system_prompt=get_managed_prompt(),
    model="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    trace_attributes={"service.name": "ResearcherAgent", "protocol": "A2A"},
    tools=[streamable_http_mcp_client],
)

if __name__ == "__main__":
    serve_a2a(
        StrandsA2AExecutor(researcher_agent, enable_a2a_compliant_streaming=True),
        host="0.0.0.0",
        port=9000,
    )
