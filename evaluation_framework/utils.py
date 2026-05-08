import boto3


def get_bedrock_prompt(prompt_arn: str) -> str:
    client = boto3.client("bedrock-agent")
    return client.get_prompt(promptIdentifier=prompt_arn)["variants"][0][
        "templateConfiguration"
    ]["text"]["text"]
    pass
