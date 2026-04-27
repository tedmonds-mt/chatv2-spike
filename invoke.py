import json
import logging
import re
import uuid

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

client = boto3.client("bedrock-agentcore")
session_id = f"user1-session-{uuid.uuid4()}"


WRITER_RUNTIME_ARN = "arn:aws:bedrock-agentcore:eu-west-2:715195480427:runtime/OrchestratorA2AClient-LMMIPXBYzq"


def extract_answer(full_response: str) -> str:
    if "```json" in full_response:
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, full_response, flags=re.DOTALL)

        if match:
            json_string = match.group(1)
            data = json.loads(json_string)
            return data.get("answer", full_response)
        else:
            return full_response
    else:
        match_exp = r"<classification_analysis>.*?</classification_analysis>"
        processed = re.sub(match_exp, "", full_response, flags=re.DOTALL)
        return processed.strip()


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
    answer = extract_answer(
        response_body.get("result", json.dumps(response_body, indent=2))
    )
    print(answer)
    return answer
