import re

import aws_cdk as cdk
import aws_cdk.aws_bedrockagentcore as bedrockagentcore
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as lambda_
import pytest
from aws_cdk.assertions import Match, Template

from infra.retrieval_stack import RetrievalStack


@pytest.fixture
def template():
    app = cdk.App(context={"aws:cdk:bundling-stacks": []})
    stack = RetrievalStack(app, "TestRetrievalStack")
    return Template.from_stack(stack)


@pytest.mark.describe("RetrievalStack")
class TestRetrievalStack:
    @pytest.mark.it("creates a lambda")
    def test_creates_lambda(self, template):
        template.has_resource_properties(
            lambda_.CfnFunction.CFN_RESOURCE_TYPE_NAME,
            {
                "FunctionName": "RetrievalLambda",
            },
        )

    @pytest.mark.it("adds an IAM role to the lambda")
    def test_lambda_has_role(self, template):
        roles = template.find_resources(iam.CfnRole.CFN_RESOURCE_TYPE_NAME)
        assert any(re.match(r"RetrievalRole[A-Z0-9]+", key) for key in roles)

    @pytest.mark.it("adds model invoke policy to lambda role")
    def test_adds_model_invoke_policy_to_lambda_role(self, template):
        expected_props = {
            "PolicyDocument": {
                "Statement": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Action": "bedrock:InvokeModel",
                                "Effect": "Allow",
                            }
                        )
                    ]
                )
            },
            "Roles": Match.array_with(
                [
                    Match.object_like(
                        {"Ref": Match.string_like_regexp(r"RetrievalRole[A-Z0-9]+")}
                    )
                ]
            ),
        }
        template.has_resource_properties(
            iam.CfnPolicy.CFN_RESOURCE_TYPE_NAME, expected_props
        )

    @pytest.mark.it("creates a gateway")
    def test_creates_gateway(self, template):
        template.resource_count_is(
            bedrockagentcore.CfnGateway.CFN_RESOURCE_TYPE_NAME, 1
        )
        gateways = template.find_resources(
            bedrockagentcore.CfnGateway.CFN_RESOURCE_TYPE_NAME
        )
        assert any(re.match(r"RetrievalGateway[A-Z0-9]+", key) for key in gateways)

    @pytest.mark.it("adds lambda target to gateway")
    def test_adds_lambda_target(self, template):
        template.has_resource_properties(
            bedrockagentcore.CfnGatewayTarget.CFN_RESOURCE_TYPE_NAME,
            {
                "TargetConfiguration": Match.object_like(
                    {
                        "Mcp": Match.object_like(
                            {
                                "Lambda": Match.object_like(
                                    {
                                        "LambdaArn": Match.object_like(
                                            {
                                                "Fn::GetAtt": Match.array_with(
                                                    [
                                                        Match.string_like_regexp(
                                                            r"RetrievalLambda[A-Z0-9]+"
                                                        ),
                                                    ]
                                                )
                                            }
                                        )
                                    }
                                )
                            }
                        )
                    }
                ),
                "GatewayIdentifier": Match.object_like(
                    {
                        "Fn::GetAtt": Match.array_with(
                            [Match.string_like_regexp(r"RetrievalGateway[A-Z0-9]+")]
                        )
                    }
                ),
            },
        )
