import gradio as gr

from invoke import invoke_agent

demo = gr.ChatInterface(
    fn=invoke_agent,
    title="Prototype",
    description="I'm a lightweight version of Gov UK chat. Ask me stuff.",
)

if __name__ == "__main__":
    demo.launch()

"""
https://bedrock-agent-runtime.eu-west-2.amazonaws.com/agents/<AGENT_ID>/agentAliases/<AGENT_ALIAS_ID>/sessions/<SESSION_ID>/text

arn:aws:bedrock-agentcore:eu-west-2:715195480427:runtime/OrchestratorA2AClient-LMMIPXBYzq/runtime-endpoint/DEFAULT
"""
