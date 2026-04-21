from aws_cdk import Duration, Stack
from aws_cdk import aws_bedrock_agentcore_alpha as agentcore
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk.aws_bedrock_agentcore_alpha import SchemaDefinitionType
from constructs import Construct


class RetrievalStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        retrieval_role = iam.Role(
            self,
            "RetrievalRole",
            assumed_by=iam.ServicePrincipal('lambda.amazonaws.com'),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    'service-role/AWSLambdaBasicExecutionRole'
                )
            ]
        )

        retrieval_role.add_to_policy(
            iam.PolicyStatement(
                actions=['bedrock:InvokeModel'],
                resources=['*']
            )
        )

        retrieval_lambda = lambda_.Function(
            self,
            "RetrievalLambda",
            function_name="RetrievalLambda",
            runtime=lambda_.Runtime.PYTHON_3_13,
            handler="handler.handler",
            role=retrieval_role,
            code=lambda_.Code.from_asset("tools/search_gov_uk"),
            timeout=Duration.seconds(30),
            environment={
                "MY_ENV_VAR": "foo"
            }
        )

        gateway = agentcore.Gateway(
            self,
            "RetrievalGateway",
            gateway_name="opensearch-gateway",
            protocol_configuration=agentcore.McpProtocolConfiguration(
                instructions="Use search_knowledge_base to retrieve relevant context.",
                search_type=agentcore.McpGatewaySearchType.SEMANTIC,
                supported_versions=[agentcore.MCPProtocolVersion.MCP_2025_06_18]
            )
        )

        gateway.add_lambda_target(
            "SearchKnowledgeBaseLambdaTarget",
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
                                type= SchemaDefinitionType.STRING,
                                description ="the search query"
                                ),
                                "top_k": agentcore.SchemaDefinition(
                                    type=SchemaDefinitionType.INTEGER,
                                    description="the top k results - default 5"
                                )
                            },
                            required=["query"]
                        )
                    )
                ]
            )
        )

