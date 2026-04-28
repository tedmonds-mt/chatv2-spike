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
        full_response = re.sub("\\n", "\n", full_response)
        pattern = r"```json.{1,2}(\{.*?\}).{1,2}```"
        match = re.search(pattern, full_response, flags=re.DOTALL)

        if match:
            json_string = match.group(1)
            try:
                data = json.loads(json_string)
            except json.JSONDecodeError:
                data = {"answer": json_string.strip()}
                print(data)

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
        accept="text/event-stream",
    )
    full_response = ""
    for line in response["response"].iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data: "):
                content = decoded_line[6:]
                try:
                    unquoted_content = json.loads(content)
                    if isinstance(unquoted_content, dict):
                        if not unquoted_content.get("artifacts"):
                            continue
                        chunks = unquoted_content.get("artifacts")[0].get("parts", [])
                        text_chunks = [
                            str(c["text"]) for c in chunks if c["kind"] == "text"
                        ]
                        unquoted_content = "".join(text_chunks)
                    full_response += unquoted_content
                except json.JSONDecodeError:
                    full_response += content.strip()

            extracted_answer = extract_answer(full_response)
            yield extracted_answer
    print(full_response)
