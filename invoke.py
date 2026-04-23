import boto3
import json
import logging
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

client = boto3.client("bedrock-agentcore")
session_id = f"user1-session-{uuid.uuid4()}"


WRITER_RUNTIME_ARN = "arn:aws:bedrock-agentcore:eu-west-1:715195480427:runtime/OrchestratorA2AClient-Kx1U9g59j8"


def invoke_agent(message: str, history: list):
    logger.info('Calling agent runtime')
    payload = {"prompt": message}
    response = client.invoke_agent_runtime(
        runtimeSessionId=session_id,
        agentRuntimeArn=WRITER_RUNTIME_ARN,
        payload=json.dumps(payload).encode("utf-8"),
    )
    response_body = json.loads(response["response"].read().decode("utf-8"))
    return response_body.get("result", response_body)


