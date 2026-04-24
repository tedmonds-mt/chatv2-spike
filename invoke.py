import boto3
import json
import logging
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

client = boto3.client("bedrock-agentcore")
session_id = f"user1-session-{uuid.uuid4()}"


WRITER_RUNTIME_ARN = "arn:aws:bedrock-agentcore:eu-west-2:715195480427:runtime/OrchestratorA2AClient-LMMIPXBYzq"


def invoke_agent(message: str, history: list):
    logger.info("Calling agent runtime")
    payload = {"prompt": message}
    response = client.invoke_agent_runtime(
        runtimeSessionId=session_id,
        agentRuntimeArn=WRITER_RUNTIME_ARN,
        payload=json.dumps(payload).encode("utf-8"),
    )
    response_body = json.loads(response["response"].read().decode("utf-8"))
    print("Response body: %s", json.dumps(response_body, indent=2))
    return response_body.get("result", json.dumps(response_body, indent=2))
