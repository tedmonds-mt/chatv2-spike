import gradio as gr
from invoke import invoke_agent

demo = gr.ChatInterface(
    fn=invoke_agent,
    title="Prototype",
    description="I'm a lightweight version of Gov UK chat. Ask me stuff.",
)

if __name__ == "__main__":
    demo.launch()
