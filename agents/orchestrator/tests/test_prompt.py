import json
import re
from json import JSONDecodeError

import pytest

from agents.orchestrator.tests.test_data.prompt_data import EVAL_DATA
from evaluation_framework.prompt_evaluation import PromptEvaluator
from evaluation_framework.types import ModelTurn, Prompt, ToolDefinition
from evaluation_framework.utils import get_bedrock_prompt

PROMPT_ARN = "arn:aws:bedrock:eu-west-2:715195480427:prompt/WIA053R63J"


@pytest.fixture(scope="module")
def tools():
    return [
        ToolDefinition(
            tool_name="StackGatewaySearchX1234567___searchGovUk",
            tool_description="Search GOV.UK knowledge base",
            input_schema={
                "query": {"type": "string", "description": "the search query"},
                "top_k": {
                    "type": "integer",
                    "description": "the top k results - default 5",
                },
            },
            required_inputs=["query"],
        ),
        ToolDefinition(
            tool_name="complex_search",
            tool_description="Delegates complex queries to the researcher agent via Bedrock AgentCore",
            input_schema={"user_input": {"type": "string"}},
            required_inputs=["user_input"],
        ),
    ]


@pytest.fixture(scope="module")
def evaluator(tools):
    prompt_text = get_bedrock_prompt(PROMPT_ARN)
    prompt = Prompt(
        id="orchestrator",
        prompt_text=prompt_text,
        model="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        tools=tools,
    )
    return PromptEvaluator(prompt)


def filter_by_test(test):
    return [t for t in EVAL_DATA if test in t.tests]


class TestOrchestrator:
    @pytest.mark.parametrize("case", filter_by_test("complexity_routing"))
    def test_complexity_routing(self, evaluator, case):
        expected_tool = case.expected_test_results["complexity_routing"]

        def evaluation(_, model_result: ModelTurn):
            used_tools = [t.tool_name for t in model_result.tool_calls]
            return 1 if len(used_tools) == 1 and expected_tool in used_tools[0] else 0

        for model_response, eval_result in evaluator.evaluate_with_function(
            case, evaluation
        ):
            assert eval_result == 1, (
                f"Case {case.id} failed. \n"
                f"Expected tool: {case.expected_test_results['complexity_routing']}\n"
                f"Actual: {[t.tool_name for t in model_response.tool_calls]}"
            )

    @pytest.mark.parametrize("case", filter_by_test("token_budget"))
    def test_token_budget(self, evaluator, case):
        token_budget = case.expected_test_results["token_budget"]

        def evaluation(_, model_result: ModelTurn):
            used_tokens = model_result.input_tokens + model_result.output_tokens
            return 1 if used_tokens <= token_budget else 0

        for model_response, eval_result in evaluator.evaluate_with_function(
            case, evaluation
        ):
            assert eval_result == 1, (
                f"Case {case.id} failed. \n"
                f"Expected max tokens: {case.expected_test_results['token_budget']}\n"
                f"Actual: {model_response.input_tokens + model_response.output_tokens}"
            )

    @pytest.mark.parametrize("case", filter_by_test("max_latency"))
    def test_max_latency(self, evaluator, case):
        max_latency = case.expected_test_results["max_latency"]

        def evaluation(_, model_result: ModelTurn):
            actual_latency = model_result.latency
            return 1 if actual_latency <= max_latency else 0

        for model_response, eval_result in evaluator.evaluate_with_function(
            case, evaluation, 3
        ):
            assert eval_result == 1, (
                f"Case {case.id} failed. \n"
                f"Expected max latency: {case.expected_test_results['max_latency']}\n"
                f"Actual: {model_response.latency}"
            )

    @pytest.mark.parametrize("case", filter_by_test("output_formatting"))
    def test_outputs_json_format(self, evaluator, case):
        def evaluation(_, model_result: ModelTurn):
            json_response = re.search("(?<=json```).*(?=```)", model_result.text)
            if not json_response:
                return {"score": 0, "reason": "No json object found"}

            try:
                loaded_response = json.loads(json_response.group())
            except JSONDecodeError:
                return {
                    "score": 0,
                    "reason": f"Not valid json: {json_response.group()}",
                }

            return (
                {"score": 1, "reason": "Answer provided"}
                if loaded_response.get("answer")
                else {"score": 0, "reason": f"No answer in {loaded_response}"}
            )

        for _, eval_result in evaluator.evaluate_with_function(case, evaluation):
            assert eval_result[0] == 1, (
                f"Case {case.id} failed. \nReason: {eval_result[1]}"
            )
