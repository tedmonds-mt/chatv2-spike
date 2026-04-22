import aws_cdk.aws_lambda as lambda_
from aws_cdk import Duration, Stack
from aws_cdk import aws_bedrock_agentcore_alpha as agentcore
from aws_cdk import aws_iam as iam
from aws_cdk.aws_bedrock_agentcore_alpha import SchemaDefinitionType
from aws_cdk.aws_lambda_python_alpha import PythonFunction
from constructs import Construct


class RetrievalStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        retrieval_role = iam.Role(
            self,
            "RetrievalRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        retrieval_role.add_to_policy(
            iam.PolicyStatement(actions=["bedrock:InvokeModel"], resources=["*"])
        )

        retrieval_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    "arn:aws:bedrock:eu-west-1::foundation-model/amazon.titan-embed-text-v2:0"
                ],
            )
        )

        retrieval_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:DescribeSecret",
                ],
                resources=[
                    "arn:aws:secretsmanager:eu-west-2:715195480427:secret:GdsChat-sUawHf"
                ],
            )
        )

        retrieval_lambda = PythonFunction(
            self,
            "RetrievalLambda",
            function_name="RetrievalLambda",
            runtime=lambda_.Runtime.PYTHON_3_13,
            entry="tools/search_gov_uk",
            handler="lambda_handler",
            index="handler.py",
            role=retrieval_role,
            timeout=Duration.seconds(30),
        )

        gateway = agentcore.Gateway(
            self,
            "RetrievalGateway",
            gateway_name="opensearch-gateway",
            protocol_configuration=agentcore.McpProtocolConfiguration(
                instructions="Use search_knowledge_base to retrieve relevant context.",
                search_type=agentcore.McpGatewaySearchType.SEMANTIC,
                supported_versions=[agentcore.MCPProtocolVersion.MCP_2025_06_18],
            ),
        )
        self.userPoolClient = gateway.user_pool_client.user_pool_client_id
        self.userPoolSecret = (
            gateway.user_pool_client.user_pool_client_secret.to_string()
        )
        self.userPoolDomain = gateway.user_pool_domain.domain_name
        self.gateway_url = gateway.gateway_url

        gateway.add_lambda_target(
            "Search",
            lambda_function=retrieval_lambda,
            description="Search knowledge base",
            tool_schema=agentcore.ToolSchema.from_inline(
                [
                    agentcore.ToolDefinition(
                        name="searchGovUk",
                        description="Search GOV.UK knowledge base",
                        input_schema=agentcore.SchemaDefinition(
                            type=SchemaDefinitionType.OBJECT,
                            properties={
                                "query": agentcore.SchemaDefinition(
                                    type=SchemaDefinitionType.STRING,
                                    description="the search query",
                                ),
                                "top_k": agentcore.SchemaDefinition(
                                    type=SchemaDefinitionType.INTEGER,
                                    description="the top k results - default 5",
                                ),
                            },
                            required=["query"],
                        ),
                    )
                ]
            ),
        )
