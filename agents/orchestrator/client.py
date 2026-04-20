import os
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
import boto3
import uuid
import json
from botocore.exceptions import ClientError

RESEARCHER_RUNTIME_ARN = os.environ.get("RESEARCHER_RUNTIME_ARN")
PROMPT_ARN = "arn:aws:bedrock:eu-west-2:281868401169:prompt/YBB8Q6KWHP:1"
REGION = os.environ.get("AWS_REGION", "eu-west-2")


def get_managed_prompt() -> str:
    """Retrieves the central prompt from Bedrock Prompt Management."""
    if not PROMPT_ARN:
        return (
            "You are a technical orchestrator. Use the 'ask_researcher' tool to gather facts."
        )
    bedrock_client = boto3.client("bedrock-agent", region_name=REGION)
    suffix = ("\n\nCRITICAL INSTRUCTION: If the 'ask_researcher' tool returns a CRITICAL SYSTEM ERROR, "
              "you must stop immediately and output the exact error text to the user. Do not write the "
              "article.")
    return bedrock_client.get_prompt(promptIdentifier=PROMPT_ARN)["variants"][0][
        "templateConfiguration"
    ]["text"]["text"] + suffix


@tool
def ask_researcher(topic: str) -> str | None:
    """Delegates deep research to the researcher agent via Bedrock AgentCore"""
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
                "parts": [{"kind": "text", "text": f"Research {topic}"}],
            }
        },
    }

    try:
        encoded_payload = json.dumps(a2a_payload).encode("utf-8")

        response = client.invoke_agent_runtime(
            agentRuntimeArn=RESEARCHER_RUNTIME_ARN, payload=encoded_payload
        )

        response_body = json.loads(response["response"].read().decode("utf-8"))
        if "error" in response_body:
            return (
                "CRITICAL SYSTEM ERROR. YOU MUST STOP AND OUTPUT THIS EXACT TEXT: "
                f"{json.dumps(response_body['error'])}"
            )

        message = response_body["result"]["message"]
        if "content" in message:
            return message["content"][0]["text"]
        elif "parts" in message:
            return message["parts"][0]["text"]
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


WRITER_SYSTEM_PROMPT = get_managed_prompt()

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    user_input = payload.get("prompt", "")
    writer_agent = Agent(
        name="WriterAgent",
        system_prompt=WRITER_SYSTEM_PROMPT,
        tools=[ask_researcher],
        model="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        trace_attributes={"service.name": "WriterAgent", "deployment": "dev"},
    )
    result = writer_agent(user_input)
    return {"result": str(result)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
