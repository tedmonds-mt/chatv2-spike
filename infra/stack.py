import string
from random import choices

from aws_cdk import Stack
from aws_cdk import (
    aws_bedrock_agentcore_alpha as agentcore_alpha,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_s3 as s3,
)
from constructs import Construct

length = 8
SUFFIX = "".join(choices(string.ascii_letters + string.digits, k=length)).lower()

BUCKET_ID = "TestBucketForChatV2GDS"
BUCKET_NAME = f"{BUCKET_ID.lower()}-{SUFFIX}"


class AgentCoreStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        cognito_domain: str,
        cognito_client: str,
        cognito_secret: str,
        mcp_url: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        researcher_runtime = agentcore_alpha.Runtime(
            self,
            "ResearcherRuntime",
            runtime_name="ResearcherA2AServer",
            agent_runtime_artifact=agentcore_alpha.AgentRuntimeArtifact.from_asset(
                "agents/researcher"
            ),
            protocol_configuration=agentcore_alpha.ProtocolType.A2A,
            environment_variables={
                "AWS_REGION": Stack.of(self).region,
                "TOKEN_URL": cognito_domain,
                "CLIENT_ID": cognito_client,
                "CLIENT_SECRET": cognito_secret,
                "MCP_URL": mcp_url,
            },
        )

        orchestrator_runtime = agentcore_alpha.Runtime(
            self,
            "OrchestratorRuntime",
            runtime_name="OrchestratorA2AClient",
            agent_runtime_artifact=agentcore_alpha.AgentRuntimeArtifact.from_asset(
                "agents/orchestrator"
            ),
            environment_variables={
                "PORT": "9000",
                "AWS_REGION": Stack.of(self).region,
                "RESEARCHER_RUNTIME_ARN": Stack.of(self).format_arn(
                    service="bedrock-agentcore",
                    resource="runtime",
                    resource_name=researcher_runtime.agent_runtime_id,
                ),
                "TOKEN_URL": cognito_domain,
                "CLIENT_ID": cognito_client,
                "CLIENT_SECRET": cognito_secret,
                "MCP_URL": mcp_url,
            },
        )

        researcher_runtime.grant_invoke(orchestrator_runtime)

        model_invoke_policy = iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=[
                "arn:aws:bedrock:*::foundation-model/*",
                "arn:aws:bedrock:*:*:inference-profile/*",
            ],
        )

        researcher_runtime.role.add_to_principal_policy(model_invoke_policy)
        orchestrator_runtime.role.add_to_principal_policy(model_invoke_policy)

        orchestrator_runtime.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["bedrock:GetPrompt"],
                resources=["*"],
            )
        )

        researcher_runtime.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["bedrock:GetPrompt"],
                resources=["*"],
            )
        )

        s3.Bucket(self, BUCKET_ID, bucket_name=BUCKET_NAME)
