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


class Extractor:
    def __init__(self):
        self.live_print = False
        self.full_text = ""
        self.full_response = ""
        self.draft = ""

    def extract_answer(self, response_chunk: str) -> str:
        self.full_text += response_chunk
        if "```json" in self.full_text:
            self.live_print = True
        if self.live_print:
            pattern1 = r".*```json\s{0,2}"
            self.draft = re.sub(pattern1, "", self.full_text, flags=re.DOTALL)
            pattern2 = r'\{\s*"answer":\s*"'
            self.draft = re.sub(pattern2, "", self.draft, flags=re.DOTALL)
            pattern3 = r'\s*",.*'
            self.draft = re.sub(pattern3, "", self.draft, flags=re.DOTALL)
        elif "<answer_preparation>" in self.full_text:
            return "<small><i>Writing answer...</i></small>"
        elif "<research_summary>" in self.full_text:
            return "<small><i>Summarising research...</i></small>"
        elif matches := re.findall(
            r"(?s)<thinking>((?:(?!<thinking>).)*?)(?:<\/thinking>|$)", self.full_text
        ):
            last_thought = matches[-1]
            if counts := re.findall(r"(?s)<count>(\d)(?:<\/count>)", self.full_text):
                last_count = 5 - int(counts[-1])
            else:
                last_count = 0

            last_thought = re.sub(r"<count>\d</count>", "", last_thought)
            last_thought = re.sub(r"<reward>[\d\.]+</reward>", "", last_thought)
            last_thought = re.sub(r"</?reflection>", "", last_thought)

            return f"<small><i>Thinking ({last_count}/5): {last_thought}</i></small>"
        else:
            return "<small><i>Searching GOV.UK...</i></small>"

        if re.search(r"```\W", self.full_text):
            self.live_print = False

        self.full_response = re.sub("\n{2,}", "\n", self.draft)
        return self.full_response

    @classmethod
    def clean_newlines(cls, input_str):
        assert isinstance(input_str, str)
        return input_str.replace("\\n", "\n")


def invoke_agent(message: str, history: list):
    logger.info("Calling agent runtime")
    payload = {"prompt": message}

    extractor = Extractor()

    response = client.invoke_agent_runtime(
        runtimeSessionId=session_id,
        agentRuntimeArn=WRITER_RUNTIME_ARN,
        payload=json.dumps(payload).encode("utf-8"),
        accept="text/event-stream",
    )
    response_chunk = ""
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
                    response_chunk = unquoted_content
                except json.JSONDecodeError:
                    response_chunk = content.strip()
            print(response_chunk)
            extracted_answer = extractor.extract_answer(response_chunk)
            cleaned = Extractor.clean_newlines(extracted_answer)
            yield cleaned
