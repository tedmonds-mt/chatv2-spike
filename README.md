# Strands & Bedrock AgentCore Tutorial

A multi-agent architecture tutorial showcasing the [Strands](https://strands.com) framework integrated with **AWS Bedrock AgentCore**. 

This application demonstrates a secure Agent-to-Agent (A2A) workflow where a **Writer Agent** (acting as a Gonzo Journalist) dynamically delegates deep-research tasks to a **Researcher Agent**. 

## 🏗️ Architecture

The project consists of three main components:
1. **Writer Agent (`agents/writer`)**: A standard Bedrock AgentCore App listening on port 8080. It dynamically fetches its system prompt from Bedrock Prompt Management and delegates factual research using the A2A protocol.
2. **Researcher Agent (`agents/researcher`)**: A Strands A2A Server listening on port 9000. It receives JSON-RPC requests from the Writer, executes the research using Claude 4.5, and returns the facts.
3. **AWS CDK Infrastructure (`infra/`)**: Deploys both agents as isolated Bedrock AgentCore Runtimes. It handles environment variables, secure networking, and IAM permissions (`bedrock:InvokeModel`, `bedrock:GetPrompt`).

## 📋 Prerequisites

* **Python 3.13+**
* [uv](https://github.com/astral-sh/uv) (for ultra-fast dependency management)
* [Poe the Poet](https://poethepoet.natn.io/) installed globally (`pip install poethepoet`)
* AWS CLI configured with credentials for your target region (e.g., `eu-west-2`)
* AWS CDK CLI installed

## 🚀 Quickstart

Use the globally installed `poe` task runner to manage the application:

### 1. Deploy the Infrastructure
If this is your first time using CDK in your AWS account/region, bootstrap it first:
```bash
poe bootstrap
```

Deploy the agents to Bedrock AgentCore:
```bash
poe deploy
```

### 2. Run the Application
Start the GUI to chat with the Orchestrator Agent:
```bash
poe app
```

## 🛠️ Development Tasks

The `pyproject.toml` includes several `poe` tasks to ensure code quality before deployment:

* `poe check`: Runs the full pre-deployment suite (lint, audit, scan, typing).
* `poe lint`: Runs Ruff to check and auto-fix linting errors.
* `poe formatter`: Formats code using Ruff.
* `poe typing`: Runs `mypy` for static type checking.
* `poe scan`: Scans the code for security vulnerabilities using `bandit`.
* `poe audit`: Audits Python dependencies for known vulnerabilities.
* `poe tests`: Runs the test suite with coverage reporting.

