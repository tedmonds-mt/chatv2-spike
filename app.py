#!/usr/bin/env python3

import aws_cdk as cdk
from infra.stack import AgentCoreStack

app = cdk.App()
AgentCoreStack(app, "StrandsA2ATutorialStack")
app.synth()

