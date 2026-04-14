import os

import uvicorn
from strands import Agent
from fastapi import FastAPI
from strands.multiagent.a2a import A2AServer

RUNTIME_URL = os.environ.get("AGENTCORE_RUNTIME_URL", "http://localhost:9000")

researcher_agent = Agent(
    name="ResearcherAgent",
    description="A specialised research agent that provides factual summaries.",
    system_prompt="You are an expert researcher. Provide concise, factual summaries on the requested topic.",
)

a2a_server = A2AServer(
    agent=researcher_agent,
    http_url=RUNTIME_URL,
    serve_at_root=True,
    enable_a2a_compliant_streaming=False
)

app = FastAPI()
app.mount("/", a2a_server.to_fastapi_app())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

