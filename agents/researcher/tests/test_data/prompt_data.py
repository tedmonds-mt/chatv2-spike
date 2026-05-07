from evaluation_framework.types import ModelTurn, PromptTestCase, ToolCall, UserTurn

EVAL_DATA = [
    PromptTestCase(
        id="direct_factual_question_1",
        description="Does it call the simple search tool for direct questions?",
        user_turn=UserTurn(text="What is the UK retirement age?"),
        expected_model_response=ModelTurn(
            tool_calls=[
                ToolCall(tool_name="simple_search", input={"search_term": "any"}),
            ],
            input_tokens=400,
            output_tokens=400,
        ),
        tests=["call_correct_tool", "token_budget"],
    ),
    PromptTestCase(
        id="simple_personal_question",
        description="Does it call the simple search tool for direct questions, even if it includes a personal detail?",
        user_turn=UserTurn(text="I am 66 - can I retire?"),
        expected_model_response=ModelTurn(
            tool_calls=[
                ToolCall(tool_name="simple_search", input={"search_term": "any"})
            ],
            latency_ms=3000,
        ),
        tests=["call_correct_tool", "latency"],
    ),
]
