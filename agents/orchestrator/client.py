import json
import logging
import os
import sys
import uuid

import boto3
import requests
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from botocore.exceptions import ClientError
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from strands import Agent, tool
from strands.tools.mcp import MCPClient

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

RESEARCHER_RUNTIME_ARN = os.environ.get("RESEARCHER_RUNTIME_ARN")
PROMPT_ARN = "arn:aws:bedrock:eu-west-2:715195480427:prompt/WIA053R63J"
REGION = os.environ.get("AWS_REGION", "eu-west-2")

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
GATEWAY_ID = os.environ.get("TOKEN_URL")
TOKEN_URL = f"https://{GATEWAY_ID}.auth.eu-west-2.amazoncognito.com/oauth2/token"
MCP_URL = os.environ.get("MCP_URL")


def get_managed_prompt() -> str:
    """Retrieves the central prompt from Bedrock Prompt Management."""
    if not PROMPT_ARN:
        return "You are a technical orchestrator. Use the 'ask_researcher' tool to gather facts."
    bedrock_client = boto3.client("bedrock-agent", region_name=REGION)
    return bedrock_client.get_prompt(promptIdentifier=PROMPT_ARN)["variants"][0][
        "templateConfiguration"
    ]["text"]["text"]


@tool
def complex_search(user_input: str) -> str | None:
    """
    Delegates complex queries to the researcher agent via Bedrock AgentCore

    Args:
        user_input (str): The user's input to the orchestrator, unchanged.

    Returns:
        str|None: The response from the researcher agent.

    """
    if not RESEARCHER_RUNTIME_ARN:
        return "Error: Researcher runtime ARN not configured."

    client = boto3.client("bedrock-agentcore", region_name=REGION)

    response_body = None

    a2a_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"kind": "text", "text": user_input}],
            }
        },
    }

    try:
        encoded_payload = json.dumps(a2a_payload).encode("utf-8")
        logging.info(f"Sending {user_input} to Bedrock AgentCore")
        response = client.invoke_agent_runtime(
            agentRuntimeArn=RESEARCHER_RUNTIME_ARN, payload=encoded_payload
        )

        logging.info(f"A2A response: {json.dumps(response)}")

        response_body = json.loads(response["response"].read().decode("utf-8"))
        if "error" in response_body:
            logging.error(f"A2A error: {json.dumps(response_body['error'])}")
            return (
                "CRITICAL SYSTEM ERROR. YOU MUST STOP AND OUTPUT THIS EXACT TEXT: "
                f"{json.dumps(response_body['error'])}"
            )

        message = response_body["result"]["message"]
        if "content" in message:
            return message["content"][0]["text"]
        elif "parts" in message:
            return message["parts"][0]["text"]
        logging.info(f"A2A message: {message}")
        return str(message)
    except (KeyError, IndexError) as e:
        print(f"The research tool threw a Key or Index Error, {e}")
        return f"CRITICAL SYSTEM ERROR. SCHEMA MISMATCH: {json.dumps(response_body)}"
    except ClientError as c:
        print(f"Boto3 Error: {c}")
        return None
    except Exception as e:
        print(f"Some other Error: {e}")
        return f"CRITICAL SYSTEM ERROR. {type(e).__name__}: {str(e)}"


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


ORCHESTRATOR_SYSTEM_PROMPT = get_managed_prompt()

app = BedrockAgentCoreApp()

mcp_access_token = fetch_access_token(CLIENT_ID, CLIENT_SECRET, TOKEN_URL)
streamable_http_mcp_client = MCPClient(
    lambda: create_streamable_http_transport(MCP_URL, mcp_access_token)
)


@app.entrypoint
def invoke(payload):
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
        result = orchestrator_agent(user_input)
    return {"result": str(result)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
