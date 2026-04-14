FROM python:3.13-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# AgentCore expects the agent server to listen on port 9000
EXPOSE 9000
CMD ["python", "server.py"]