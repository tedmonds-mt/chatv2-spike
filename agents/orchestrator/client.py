import logging
import os
import sys
import uuid
from typing import Any, AsyncGenerator
from urllib.parse import quote

import boto3
import httpx
import requests
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from strands import Agent, tool
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


RESEARCHER_RUNTIME_ARN = get_env("RESEARCHER_RUNTIME_ARN")
CLIENT_ID = get_env("CLIENT_ID")
CLIENT_SECRET = get_env("CLIENT_SECRET")
GATEWAY_ID = get_env("TOKEN_URL")
MCP_URL = get_env("MCP_URL")
REGION = os.environ.get("AWS_REGION", "eu-west-2")

A2A_POOL_ID = get_env("A2A_POOL_ID")
A2A_POOL_CLIENT = get_env("A2A_POOL_CLIENT")
A2A_POOL_SECRET = get_env("A2A_POOL_SECRET")
A2A_DOMAIN_PREFIX = get_env("A2A_POOL_DOMAIN")

TOKEN_URL = f"https://{GATEWAY_ID}.auth.{REGION}.amazoncognito.com/oauth2/token"
A2A_POOL_URL = (
    f"https://{A2A_DOMAIN_PREFIX}.auth.{REGION}.amazoncognito.com/oauth2/token"
)

PROMPT_ARN = "arn:aws:bedrock:eu-west-2:715195480427:prompt/WIA053R63J"
session = boto3.Session(region_name=REGION)


def get_managed_prompt() -> str:
    """Retrieves the central prompt from Bedrock Prompt Management."""
    if not PROMPT_ARN:
        return "You are a technical orchestrator. Use the 'complex_search' tool to gather facts."
    bedrock_client = boto3.client("bedrock-agent", region_name=REGION)
    return bedrock_client.get_prompt(promptIdentifier=PROMPT_ARN)["variants"][0][
        "templateConfiguration"
    ]["text"]["text"]


def fetch_access_token(client_id, client_secret, token_url):
    """Gets access token from cognito"""
    response = requests.post(
        token_url,
        data=f"grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.error(f"Cognito Auth Error: {response.text}")
        raise e

    return response.json()["access_token"]


def create_streamable_http_transport(mcp_url: str, access_token: str):
    client = create_mcp_http_client(headers={"Authorization": f"Bearer {access_token}"})
    return streamable_http_client(mcp_url, http_client=client)


def create_message(*, role: Role = Role.user, text: str) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid.uuid4().hex,
    )


@tool
async def complex_search(
    user_input: str,
) -> AsyncGenerator[
    str
    | Message
    | Task
    | tuple[Task, TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None],
    Any,
]:
    """
    Delegates complex queries to the researcher agent via Bedrock AgentCore
    """
    escaped_agent_arn = quote(RESEARCHER_RUNTIME_ARN, safe="")
    runtime_url = f"https://bedrock-agentcore.{REGION}.amazonaws.com/runtimes/{escaped_agent_arn}/invocations"

    bearer_token = fetch_access_token(A2A_POOL_CLIENT, A2A_POOL_SECRET, A2A_POOL_URL)

    session_id = str(uuid.uuid4())
    print(f"Generated session ID: {session_id}")
    yield "Researcher is researching"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
    }

    async with httpx.AsyncClient(timeout=500, headers=headers) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=runtime_url)
        agent_card = await resolver.get_agent_card()

        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=True,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = create_message(text=user_input)

        async for event in client.send_message(msg):
            if isinstance(event, Message):
                logging.info(event.model_dump_json(exclude_none=True, indent=2))
                yield event
            elif isinstance(event, tuple) and len(event) == 2:
                task, update_event = event
                logging.info(
                    f"Task: {task.model_dump_json(exclude_none=True, indent=2)}"
                )
                if update_event:
                    logging.info(
                        f"Update: {update_event.model_dump_json(exclude_none=True, indent=2)}"
                    )
                    yield task
            else:
                logging.info(f"Response: {str(event)}")
                yield event


ORCHESTRATOR_SYSTEM_PROMPT = get_managed_prompt()

app = BedrockAgentCoreApp()

mcp_access_token = fetch_access_token(CLIENT_ID, CLIENT_SECRET, TOKEN_URL)
streamable_http_mcp_client = MCPClient(
    lambda: create_streamable_http_transport(MCP_URL, mcp_access_token)
)


@app.entrypoint
async def invoke(payload):
    user_input = payload.get("prompt", "")
    with streamable_http_mcp_client:
        mcp_tools = streamable_http_mcp_client.list_tools_sync()
        orchestrator_agent = Agent(
            name="OrchestratorAgent",
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            tools=mcp_tools + [complex_search],
            model="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
            trace_attributes={"service.name": "OrchestratorAgent", "deployment": "dev"},
        )

        stream = orchestrator_agent.stream_async(user_input)
        previous_tool_response = ""
        async for event in stream:
            logging.info(f"Raw event: {event}")
            if tool_stream := event.get("tool_stream_event"):
                if update := tool_stream.get("data"):
                    try:
                        full_agent_response = update.artifacts.parts
                        update.artifacts.parts = full_agent_response[
                            len(previous_tool_response) :
                        ]
                        previous_tool_response = full_agent_response
                        yield update
                    except Exception:
                        yield update
            elif "data" in event and isinstance(event["data"], str):
                yield event["data"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
