from aws_cdk import Stack
from aws_cdk import (
    aws_bedrock_agentcore_alpha as agentcore_alpha,
)
from aws_cdk import (
    aws_cognito as cognito,
)
from aws_cdk import (
    aws_iam as iam,
)
from constructs import Construct

length = 8
SUFFIX = "dev-env-1"

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

        memory_role = iam.Role(
            self,
            "MemoryExecutionRole",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonBedrockAgentCoreMemoryBedrockModelInferenceExecutionRolePolicy"
                )
            ],
        )

        shared_memory = agentcore_alpha.Memory(
            self,
            "SharedMemory",
            memory_name="shared_memory",
            description="Shared memory for Orchestrator and Researcher",
            execution_role=memory_role,
            memory_strategies=[
                agentcore_alpha.MemoryStrategy.using_semantic(
                    namespaces="/strategies/{memoryStrategyId}/actions/{actionId}/sessions/{sessionId}",
                    name="semantic",
                )
            ],
        )

        access_auth = cognito.UserPool(self, "A2AAccess")

        a2a_resource_server = access_auth.add_resource_server(
            "A2AResourceServer",
            identifier="a2a",
            scopes=[
                cognito.ResourceServerScope(
                    scope_name="invoke",
                    scope_description="Invoke the AgentCore Researcher",
                )
            ],
        )

        access_auth_client = cognito.UserPoolClient(
            self,
            "A2AClient",
            user_pool=access_auth,
            generate_secret=True,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(client_credentials=True),
                scopes=[cognito.OAuthScope.custom("a2a/invoke")],
            ),
        )
        access_auth_client.node.add_dependency(a2a_resource_server)

        a2a_domain = access_auth.add_domain(
            "A2ADomain",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix=f"a2a-researcher-{SUFFIX}"
            ),
        )

        researcher_runtime = agentcore_alpha.Runtime(
            self,
            "ResearcherRuntime",
            runtime_name="ResearcherA2AServer",
            authorizer_configuration=agentcore_alpha.RuntimeAuthorizerConfiguration.using_cognito(
                user_pool=access_auth, user_pool_clients=[access_auth_client]
            ),
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
                "MEMORY_ID": shared_memory.memory_id,
                "A2A_POOL_DOMAIN": a2a_domain.domain_name,
                "A2A_POOL_ID": access_auth.user_pool_id,
                "A2A_POOL_CLIENT": access_auth_client.user_pool_client_id,
                "A2A_POOL_SECRET": access_auth_client.user_pool_client_secret.to_string(),
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

        orchestrator_runtime.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock-agentcore:ListEvents",
                    "bedrock-agentcore:CreateEvent",
                    "bedrock-agentcore:RetrieveMemoryRecords",
                ],
                resources=[shared_memory.memory_arn],
            )
        )

        researcher_runtime.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["bedrock:GetPrompt"],
                resources=["*"],
            )
        )

        # s3.Bucket(self, BUCKET_ID, bucket_name=BUCKET_NAME)
