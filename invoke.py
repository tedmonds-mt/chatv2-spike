import boto3
import json
import uuid

WRITER_RUNTIME_ARN = 'arn:aws:bedrock-agentcore:eu-west-2:281868401169:runtime/WriterA2AClient-eO3j7LFNX6'


def create_a2a_payload(prompt: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": prompt}],
            }
        },
    }


def invoke_bedrock_a2a_agent(runtime_arn: str, prompt: str) -> str:
    client = boto3.client('bedrock-agentcore')
    payload = create_a2a_payload(prompt)

    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        payload=json.dumps(payload)
    )
    response_body = json.loads(response["payload"].read().decode("utf-8"))

    try:
        return response_body["result"]["message"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return f"Raw Response {json.dumps(response_body, indent=2)}"


def main():
    print("🤖 Welcome to the AgentCore Interactive CLI")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            topic = input("Enter a topic for the Writer Agent: ").strip()

            if topic.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            if not topic:
                continue

            print(f"\n⏳ Delegating task to Writer Agent...")

            prompt = f"The topic is: {topic}"
            result = invoke_bedrock_a2a_agent(WRITER_RUNTIME_ARN, prompt)

            print("\n--- Final Output ---")
            print(result)
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error invoking agent: {e}\n")


if __name__ == "__main__":
    main()



