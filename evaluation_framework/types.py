from typing import Any, List, Optional

from pydantic import BaseModel


class ToolCall(BaseModel):
    tool_use_id: Optional[str]
    tool_name: str
    input: dict[str, Any]


class ToolResult(BaseModel):
    tool_use_id: str
    tool_name: str
    response_json: dict[str, Any]


class UserTurn(BaseModel):
    text: str
    tool_responses: Optional[List[ToolResult]] = None


class ModelTurn(BaseModel):
    text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class PromptTestCase(BaseModel):
    id: str
    description: Optional[str] = None
    user_turn: UserTurn
    conversation_history: Optional[List[UserTurn | ModelTurn]] = None
    actual_model_responses: Optional[List[ModelTurn]] = None
    expected_model_response: ModelTurn
    tests: List[str]


class Prompt(BaseModel):
    id: str
    prompt_text: Optional[str] = None
    prompt_arn: Optional[str] = None
    evals: Optional[List[PromptTestCase]] = None
