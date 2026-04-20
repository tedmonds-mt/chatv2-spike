from aws_cdk import Stack
import aws_cdk.aws_bedrock_agentcore_alpha as agentcore_alpha
from constructs import Construct
from aws_cdk import aws_iam as iam


class AgentCoreStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        researcher_runtime = agentcore_alpha.Runtime(
            self,
            "ResearcherRuntime",
            runtime_name="ResearcherA2AServer",
            agent_runtime_artifact=agentcore_alpha.AgentRuntimeArtifact.from_asset(
                "agents/researcher"
            ),
            protocol_configuration=agentcore_alpha.ProtocolType.A2A,
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
                resources=["*"],  # Allow it to retrieve your managed prompt ARN
            )
        )
