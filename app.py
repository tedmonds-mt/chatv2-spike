import aws_cdk as cdk

from infra.retrieval_stack import RetrievalStack
from infra.stack import AgentCoreStack

app = cdk.App()
AgentCoreStack(app, "StrandsA2ATutorialStack")
RetrievalStack(app, "ToolStack")
app.synth()
