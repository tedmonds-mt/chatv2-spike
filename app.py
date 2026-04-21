import aws_cdk as cdk

from infra.retrieval_stack import RetrievalStack
from infra.stack import AgentCoreStack

app = cdk.App()
r = RetrievalStack(app, "ToolStack")
AgentCoreStack(
    app,
    "StrandsA2ATutorialStack",
    cognito_domain=r.userPoolDomain,
    cognito_client=r.userPoolClient,
    cognito_secret=r.userPoolSecret,
    mcp_url=r.gateway_url,
)
app.synth()
