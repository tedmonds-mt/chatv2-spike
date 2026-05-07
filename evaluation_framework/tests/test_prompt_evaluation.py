from unittest.mock import MagicMock, patch

import pytest

from evaluation_framework.prompt_evaluation import PromptEvaluator
from evaluation_framework.types import (
    ModelTurn,
    Prompt,
    PromptTestCase,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UserTurn,
)


@pytest.fixture
def prompt_evaluator():
    prompt = Prompt(id="id_123", prompt_text="Some prompt text", model="model")
    return PromptEvaluator(
        prompt=prompt,
    )


@pytest.fixture
def test_case():
    return PromptTestCase(
        id="id_123",
        conversation_history=[UserTurn(text="I am a message from a user")],
        expected_model_response=ModelTurn(text="I am a message from the model"),
        tests=["test_category"],
    )


@pytest.fixture
def llm_response():
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "I am a text response"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 400,
            "outputTokens": 900,
            "totalTokens": 1300,
        },
        "metrics": {
            "latencyMs": 100,
        },
    }


@patch(
    "evaluation_framework.prompt_evaluation.PromptEvaluator.parse_conversation_history",
)
@pytest.mark.describe("PromptEvaluator runs prompt")
class TestRunsPrompt:
    @pytest.mark.it("adds simple text response from model to model turn")
    def test_adds_single_response_text(
        self, _, prompt_evaluator, test_case, llm_response
    ):
        with patch.object(PromptEvaluator, "bedrock_client") as mock_client:
            mock_client.converse.return_value = llm_response
            model_response = prompt_evaluator.run_prompt(test_case)

        assert model_response.text == "I am a text response"
        assert model_response.tool_calls is None
        assert model_response.input_tokens == 400
        assert model_response.output_tokens == 900
        assert model_response.latency == 100

    def test_adds_multiple_response_texts(
        self, _, prompt_evaluator, test_case, llm_response
    ):
        llm_response["output"]["message"]["content"].append(
            {"text": "I am additional content"}
        )
        with patch.object(PromptEvaluator, "bedrock_client") as mock_client:
            mock_client.converse.return_value = llm_response
            model_response = prompt_evaluator.run_prompt(test_case)

        assert model_response.text == "I am a text response;I am additional content"

    def test_adds_single_tool_call(self, _, prompt_evaluator, test_case, llm_response):
        llm_response["output"]["message"]["content"].append(
            {
                "toolUse": {
                    "toolUseId": "12cd",
                    "name": "some_tool",
                    "input": {"arg1": 1, "arg2": "value2"},
                    "type": "server_tool_use",
                }
            }
        )
        with patch.object(PromptEvaluator, "bedrock_client") as mock_client:
            mock_client.converse.return_value = llm_response
            model_response = prompt_evaluator.run_prompt(test_case)

        assert len(model_response.tool_calls) == 1
        assert model_response.tool_calls[0].tool_name == "some_tool"

    def test_adds_multiple_tool_calls(
        self, _, prompt_evaluator, test_case, llm_response
    ):
        llm_response["output"]["message"]["content"].append(
            {
                "toolUse": {
                    "toolUseId": "12cd",
                    "name": "some_tool",
                    "input": {"arg1": 1, "arg2": "value2"},
                    "type": "server_tool_use",
                }
            }
        )
        llm_response["output"]["message"]["content"].append(
            {
                "toolUse": {
                    "toolUseId": "34ab",
                    "name": "another_tool",
                    "input": {"argA": ["1", "2"], "argB": {"foo": "bar"}},
                    "type": "server_tool_use",
                }
            }
        )
        with patch.object(PromptEvaluator, "bedrock_client") as mock_client:
            mock_client.converse.return_value = llm_response
            model_response = prompt_evaluator.run_prompt(test_case)

        assert len(model_response.tool_calls) == 2
        assert model_response.tool_calls[0].tool_name == "some_tool"
        assert model_response.tool_calls[1].tool_name == "another_tool"
        assert all(
            k in model_response.tool_calls[1].input.keys() for k in ["argA", "argB"]
        )

    def test_adds_result_to_test_case(
        self, _, prompt_evaluator, test_case, llm_response
    ):
        assert len(test_case.actual_model_responses) == 0

        with patch.object(PromptEvaluator, "bedrock_client") as mock_client:
            mock_client.converse.return_value = llm_response
            model_response = prompt_evaluator.run_prompt(test_case)

        assert len(test_case.actual_model_responses) == 1
        assert test_case.actual_model_responses[0] == model_response


class TestParseTool:
    def test_parses_simple_tool(self):
        simple_tool = ToolDefinition(
            tool_name="simple_tool",
            tool_description="I am a tool to do a simple task",
            input_schema={"arg1": "int", "arg2": "string"},
            required_inputs=["arg1"],
        )

        tool_definition = PromptEvaluator.parse_tool(simple_tool)

        assert all(
            [
                k in tool_definition.keys()
                for k in ["name", "description", "input_schema"]
            ]
        )
        assert tool_definition["input_schema"]["json"]["required"] == ["arg1"]

    def test_fails_when_required_inputs_mismatch(self):
        mismatched_tool = ToolDefinition(
            tool_name="mismatched_tool",
            tool_description="I am a tool with a required arg not in the input schema",
            input_schema={"arg1": "int", "arg2": "string"},
            required_inputs=["arg3"],
        )
        with pytest.raises(KeyError):
            PromptEvaluator.parse_tool(mismatched_tool)


@pytest.fixture
def test_case_non_conversation_history_kwargs():
    return {
        "id": "test_case_with_full_history",
        "expected_model_response": ModelTurn(
            text="I am another message from the model"
        ),
        "tests": ["test_category"],
    }


class TestParseConversationHistory:
    def test_parses_full_conversation_history(
        self, test_case_non_conversation_history_kwargs
    ):
        test_case = PromptTestCase(
            conversation_history=[
                UserTurn(text="I am a message from a user"),
                ModelTurn(text="I am a message from the model"),
                UserTurn(text="I am another message from the user"),
            ],
            **test_case_non_conversation_history_kwargs,
        )

        conversation_history = PromptEvaluator.parse_conversation_history(
            test_case.conversation_history
        )

        assert len(conversation_history) == 3
        assert conversation_history[0]["role"] == "user"
        assert conversation_history[1]["role"] == "assistant"
        assert conversation_history[1]["content"] == [
            {"text": "I am a message from the model"}
        ]

    def test_parses_with_just_user_message(
        self, test_case_non_conversation_history_kwargs
    ):
        test_case = PromptTestCase(
            conversation_history=[UserTurn(text="I am a first message from a user")],
            **test_case_non_conversation_history_kwargs,
        )

        conversation_history = PromptEvaluator.parse_conversation_history(
            test_case.conversation_history
        )

        assert len(conversation_history) == 1
        assert conversation_history[0]["role"] == "user"
        assert conversation_history[0]["content"] == [
            {"text": "I am a first message from a user"}
        ]

    def test_parses_model_tool_call_and_response(
        self, test_case_non_conversation_history_kwargs
    ):
        test_case = PromptTestCase(
            conversation_history=[
                UserTurn(text="I am a user requesting a tool call"),
                ModelTurn(
                    text="I will call a tool",
                    tool_calls=[
                        ToolCall(
                            tool_use_id="123",
                            tool_name="some_tool",
                            input={"arg1": "value1"},
                        )
                    ],
                ),
                UserTurn(
                    tool_responses=[
                        ToolResult(
                            tool_use_id="123",
                            tool_name="some_tool",
                            response='{"some": "response"}',
                        )
                    ]
                ),
            ],
            **test_case_non_conversation_history_kwargs,
        )

        conversation_history = PromptEvaluator.parse_conversation_history(
            test_case.conversation_history
        )

        assert len(conversation_history) == 3
        assert conversation_history[1]["role"] == "assistant"
        assert conversation_history[2]["role"] == "user"
        assert len(conversation_history[1]["content"]) == 2
        assert conversation_history[1]["content"][-1]["toolUse"]["name"] == "some_tool"

    def test_fails_without_alternating_model_user_turns(
        self, test_case_non_conversation_history_kwargs
    ):
        test_case = PromptTestCase(
            conversation_history=[
                UserTurn(text="I am a message from a user"),
                UserTurn(text="I am another message from the user"),
            ],
            **test_case_non_conversation_history_kwargs,
        )

        with pytest.raises(TypeError):
            PromptEvaluator.parse_conversation_history(test_case.conversation_history)


@pytest.fixture
def result_evaluator():
    mock = MagicMock()
    mock.side_effect = range(4)
    return mock


class TestEvaluateWithFunction:
    def test_runs_prompt_if_no_previous_results(
        self, prompt_evaluator, result_evaluator, test_case
    ):
        assert not test_case.actual_model_responses

        def run_prompt_side_effect(input_test_case):
            input_test_case.actual_model_responses.append("model response")

        with patch.object(prompt_evaluator, "run_prompt") as mock_run_prompt:
            mock_run_prompt.side_effect = run_prompt_side_effect
            results = prompt_evaluator.evaluate_with_function(
                test_case=test_case, evaluator=result_evaluator
            )

        mock_run_prompt.assert_called_once()
        assert len(results) == 1
        assert results[0] == ("model response", 0)

    def test_uses_previous_results(self, prompt_evaluator, result_evaluator, test_case):
        test_case.actual_model_responses = [
            ModelTurn(text="first response"),
            ModelTurn(text="second response"),
        ]

        with patch.object(prompt_evaluator, "run_prompt") as mock_run_prompt:
            results = prompt_evaluator.evaluate_with_function(
                test_case=test_case, evaluator=result_evaluator, repeats=2
            )

        mock_run_prompt.assert_not_called()
        assert len(results) == 2

    def test_only_evaluates_repeats_results(
        self, prompt_evaluator, result_evaluator, test_case
    ):
        test_case.actual_model_responses = [
            ModelTurn(text="first response"),
            ModelTurn(text="second response"),
            ModelTurn(text="extra response"),
        ]

        with patch.object(prompt_evaluator, "run_prompt") as mock_run_prompt:
            results = prompt_evaluator.evaluate_with_function(
                test_case=test_case, evaluator=result_evaluator, repeats=2
            )

        mock_run_prompt.assert_not_called()
        assert len(results) == 2
        assert results[0] == (ModelTurn(text="first response"), 0)
        assert results[1] == (ModelTurn(text="second response"), 1)


@pytest.mark.describe("PromptEvaluator evaluates with judge")
class TestEvaluateWithJudge:
    @patch.object(PromptEvaluator, "evaluate_with_function")
    @pytest.mark.it("delegates to evaluate_with_function with correct arguments")
    def test_delegates_to_evaluate_with_function(
        self, mock_eval_func, prompt_evaluator, test_case
    ):
        prompt_evaluator.evaluate_with_judge(
            test_case=test_case, success_criteria="Must be helpful", repeats=3
        )

        mock_eval_func.assert_called_once()
        kwargs = mock_eval_func.call_args.kwargs
        assert kwargs["test_case"] == test_case
        assert kwargs["repeats"] == 3
        assert callable(kwargs["evaluator"])  # Verifies the inner function was passed

    @patch.object(PromptEvaluator, "evaluate_with_function")
    @pytest.mark.it("inner judge uses self.model when eval_model is None")
    def test_inner_judge_uses_default_model(
        self, mock_eval_func, prompt_evaluator, test_case
    ):
        prompt_evaluator.evaluate_with_judge(
            test_case, success_criteria="Must be helpful"
        )
        inner_judge = mock_eval_func.call_args.kwargs["evaluator"]

        with patch.object(PromptEvaluator, "bedrock_client") as mock_client:
            mock_client.converse.return_value = {
                "output": {
                    "message": {"content": [{"text": '{"score": 1, "reason": "Good"}'}]}
                }
            }

            inner_judge(test_case, ModelTurn(text="A helpful response"))

        call_kwargs = mock_client.converse.call_args.kwargs
        assert call_kwargs["modelId"] == prompt_evaluator.model

    @patch.object(PromptEvaluator, "evaluate_with_function")
    @pytest.mark.it("inner judge uses explicitly provided eval_model")
    def test_inner_judge_uses_provided_model(
        self, mock_eval_func, prompt_evaluator, test_case
    ):
        prompt_evaluator.evaluate_with_judge(
            test_case, success_criteria="Must be helpful", eval_model="claude-3-opus"
        )
        inner_judge = mock_eval_func.call_args.kwargs["evaluator"]

        with patch.object(PromptEvaluator, "bedrock_client") as mock_client:
            mock_client.converse.return_value = {
                "output": {
                    "message": {"content": [{"text": '{"score": 1, "reason": "Good"}'}]}
                }
            }

            inner_judge(test_case, ModelTurn(text="A helpful response"))

        call_kwargs = mock_client.converse.call_args.kwargs
        assert call_kwargs["modelId"] == "claude-3-opus"

    @patch.object(PromptEvaluator, "evaluate_with_function")
    @pytest.mark.it("inner judge correctly constructs the prompt and parses valid JSON")
    def test_inner_judge_parses_json_response(
        self, mock_eval_func, prompt_evaluator, test_case
    ):
        prompt_evaluator.evaluate_with_judge(
            test_case, success_criteria="Must be polite"
        )
        inner_judge = mock_eval_func.call_args.kwargs["evaluator"]

        with patch.object(PromptEvaluator, "bedrock_client") as mock_client:
            mock_client.converse.return_value = {
                "output": {
                    "message": {"content": [{"text": '{"score": 0, "reason": "Rude"}'}]}
                }
            }

            result = inner_judge(test_case, ModelTurn(text="Go away"))

        assert result == {"score": 0, "reason": "Rude"}

        call_kwargs = mock_client.converse.call_args.kwargs
        assert call_kwargs["inferenceConfig"]["stopSequences"] == ["```"]

        messages = call_kwargs["messages"]
        assert messages[-1]["role"] == "assistant"
        assert "```json" in messages[-1]["content"][0]["text"]
