import json
from typing import Any, Callable

import boto3

from evaluation_framework.types import (
    ModelTurn,
    Prompt,
    PromptTestCase,
    ToolCall,
    ToolDefinition,
    UserTurn,
)


class PromptEvaluator:
    bedrock_client = boto3.client("bedrock-runtime", region_name="eu-west-2")

    def __init__(self, prompt: Prompt):
        self.prompt = prompt
        self.model = prompt.model
        if self.prompt.tools:
            self.tool_config = {
                "tools": [self.parse_tool(t) for t in self.prompt.tools],
                "toolChoice": {"any": {}},
            }
        else:
            self.tool_config = None

    def evaluate_with_judge(
        self,
        test_case: PromptTestCase,
        success_criteria: str,
        eval_model: str | None = None,
        repeats: int = 1,
    ) -> list[tuple[ModelTurn, Any]]:
        if not eval_model:
            eval_model = self.model

        def eval_with_llm_judge(
            input_test_case: PromptTestCase,
            model_output: ModelTurn,
        ) -> dict[str, Any]:
            prompt = f"""
            Evaluate the output of an agent based on the following:
        
        LAST TURN: {input_test_case.conversation_history[-1].model_dump()}\n
        LLM OUTPUT: {model_output.model_dump()}
        
        SUCCESS CRITERIA: {success_criteria}
        
        Score the response either 1 if it met the success criteria, or 0 otherwise.
        Also provide a brief reason for the score.
        
        Output JSON format: {{"score": int, "reason": "string"}}
            """
            eval_conversation = [UserTurn(text=prompt), ModelTurn(text="```json")]
            response = self.bedrock_client.converse(
                modelId=eval_model,
                messages=self.parse_conversation_history(eval_conversation),
                inferenceConfig={"stopSequences": ["```"]},
            )
            response_content = response["output"]["message"]["content"][0]["text"]
            clean_content = response_content.rstrip("`").strip()
            return json.loads(clean_content)

        return self.evaluate_with_function(
            test_case=test_case, evaluator=eval_with_llm_judge, repeats=repeats
        )

    def evaluate_with_function(
        self, test_case: PromptTestCase, evaluator: Callable, repeats: int = 1
    ) -> list[tuple[ModelTurn, Any]]:
        for _ in range(repeats - len(test_case.actual_model_responses)):
            self.run_prompt(test_case)

        responses_to_evaluate = test_case.actual_model_responses[:repeats]
        results = [
            (response, evaluator(test_case, response))
            for response in responses_to_evaluate
        ]

        return results

    @staticmethod
    def parse_tool(tool: ToolDefinition) -> dict[str, Any]:
        if any(k not in tool.input_schema.keys() for k in tool.required_inputs):
            raise KeyError("All required inputs must be in the input schema")

        tool_spec = {
            "name": tool.tool_name,
            "description": tool.tool_description,
            "input_schema": {
                "json": {
                    "type": "object",
                    "properties": tool.input_schema,
                    "required": tool.required_inputs,
                }
            },
        }
        return tool_spec

    @staticmethod
    def parse_conversation_history(
        conversation_history: list[UserTurn | ModelTurn],
    ) -> list[dict[str, Any]]:
        if not all(
            t.__class__.__name__ in ["UserTurn", "ModelTurn"]
            for t in conversation_history
        ):
            raise TypeError(
                "All parts of the conversation history must be UserTurn or ModelTurn"
            )

        if not all(
            type(a) is not type(b)
            for a, b in zip(conversation_history, conversation_history[1:])
        ):
            raise TypeError("User and model turns must alternate")

        conversation = []
        for turn in conversation_history:
            parts = []
            if turn.text:
                parts.append({"text": turn.text})
            if turn.__class__.__name__ == "UserTurn":
                role = "user"
                if turn.tool_responses:
                    for response in turn.tool_responses:
                        parts.append(
                            {
                                "toolResult": {
                                    "toolUseId": response.tool_use_id,
                                    "name": response.tool_name,
                                    "content": [
                                        {
                                            "json": json.loads(response.response),
                                            "text": response.response,
                                        }
                                    ],
                                }
                            }
                        )
            elif turn.__class__.__name__ == "ModelTurn":
                role = "assistant"
                if turn.tool_calls:
                    for call in turn.tool_calls:
                        parts.append(
                            {
                                "toolUse": {
                                    "toolUseId": call.tool_use_id,
                                    "name": call.tool_name,
                                    "input": call.input,
                                    "type": "server_tool_use",
                                }
                            }
                        )
            else:
                raise TypeError(
                    "All parts of the conversation history must be UserTurn or ModelTurn"
                )
            conversation.append({"content": parts, "role": role})
        return conversation

    def run_prompt(self, test_case: PromptTestCase) -> ModelTurn:
        response = self.bedrock_client.converse(
            modelId=self.model,
            system=[{"text": self.prompt.prompt_text}],
            messages=self.parse_conversation_history(test_case.conversation_history),
            toolConfig=self.tool_config,
        )
        response_content = response["output"]["message"]["content"]

        text_parts = [p["text"] for p in response_content if p.get("text")]
        tool_use_parts = [
            ToolCall(
                tool_use_id=p["toolUse"]["toolUseId"],
                tool_name=p["toolUse"]["name"],
                input=p["toolUse"]["input"],
            )
            for p in response_content
            if p.get("toolUse")
        ]

        model_turn = ModelTurn(
            text=";".join(text_parts),
            tool_calls=tool_use_parts or None,
            input_tokens=response["usage"]["inputTokens"],
            output_tokens=response["usage"]["outputTokens"],
            latency=response["metrics"]["latencyMs"],
        )

        test_case.actual_model_responses.append(model_turn)

        return model_turn
