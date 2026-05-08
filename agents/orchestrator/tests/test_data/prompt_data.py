from evaluation_framework.types import PromptTestCase, UserTurn

EVAL_DATA = [
    PromptTestCase(
        id="direct_factual_question_1",
        description="Does it call the simple search tool for direct questions?",
        conversation_history=[UserTurn(text="What is the UK retirement age?")],
        expected_test_results={
            "complexity_routing": "complexSearch",
            "token_budget": 4000,
        },
        tests=["complexity_routing", "token_budget"],
    ),
    PromptTestCase(
        id="simple_personal_question_1",
        description="Does it call the simple search tool for direct questions, even if it includes a personal detail?",
        conversation_history=[UserTurn(text="I am 66 - can I retire?")],
        expected_test_results={
            "complexity_routing": "searchGovUk",
            "max_latency": 2000,
        },
        tests=["complexity_routing", "max_latency"],
    ),
    PromptTestCase(
        id="simple_greeting_1",
        description="User says hello",
        conversation_history=[UserTurn(text="Hello")],
        tests=["output_formatting"],
    ),
]
