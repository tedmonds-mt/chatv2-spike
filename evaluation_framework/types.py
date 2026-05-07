from typing import Any, List, Optional

from pydantic import BaseModel


class ToolCall(BaseModel):
    tool_use_id: Optional[str] = None
    tool_name: str
    input: dict[str, Any]


class ToolResult(BaseModel):
    tool_use_id: str
    tool_name: str
    response: str


class UserTurn(BaseModel):
    text: str = None
    tool_responses: Optional[List[ToolResult]] = None


class ModelTurn(BaseModel):
    text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency: Optional[int] = None


class PromptTestCase(BaseModel):
    id: str
    description: Optional[str] = None
    conversation_history: Optional[List[UserTurn | ModelTurn]]
    actual_model_responses: Optional[List[ModelTurn]] = []
    expected_model_response: ModelTurn
    tests: List[str]


class ToolDefinition(BaseModel):
    tool_name: str
    tool_description: str
    input_schema: dict[str, Any]
    required_inputs: List[str]


class Prompt(BaseModel):
    id: str
    prompt_text: Optional[str] = None
    prompt_arn: Optional[str] = None
    model: str
    tools: Optional[List[ToolDefinition]] = None
