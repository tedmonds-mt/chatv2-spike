import boto3
import json

WRITER_RUNTIME_ARN = "arn:aws:bedrock-agentcore:eu-west-2:281868401169:runtime/WriterA2AClient-eO3j7LFNX6"


def main():
    print("🤖 Welcome to the AgentCore Interactive CLI")
    print("Type 'exit' or 'quit' to stop.\n")

    client = boto3.client("bedrock-agentcore")

    while True:
        try:
            topic = input("Enter a topic for the Writer Agent: ").strip()

            if topic.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not topic:
                continue

            print("\n⏳ Delegating task to Writer Agent...")

            payload = {"prompt": f"The topic is: {topic}"}
            response = client.invoke_agent_runtime(
                agentRuntimeArn=WRITER_RUNTIME_ARN,
                payload=json.dumps(payload).encode("utf-8"),
            )

            response_body = json.loads(response["response"].read().decode("utf-8"))

            print("\n--- Final Output ---")
            print(response_body.get("result", response_body))
            print("-" * 40 + "\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error invoking agent: {e}\n")


if __name__ == "__main__":
    main()
