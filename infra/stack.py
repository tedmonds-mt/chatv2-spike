from aws_cdk import Stack
import aws_cdk.aws_bedrock_agentcore_alpha as agentcore_alpha
from constructs import Construct


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
            environment_variables={"PORT": "9000"},
        )

        writer_runtime = agentcore_alpha.Runtime(
            self,
            "WriterRuntime",
            runtime_name="WriterA2AClient",
            agent_runtime_artifact=agentcore_alpha.AgentRuntimeArtifact.from_asset(
                "agents/writer"
            ),
            environment_variables={
                "RESEARCHER_RUNTIME_ARN": Stack.of(self).format_arn(
                    service="bedrock-agentcore",
                    resource="runtime",
                    resource_name=researcher_runtime.agent_runtime_id
                )
            },
        )

        researcher_runtime.grant_invoke(writer_runtime)
