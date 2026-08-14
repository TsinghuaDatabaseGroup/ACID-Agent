from .smolagents_lib import Tool
from .smolagents_lib.models import Model


class AnswerInspectorTool(Tool):
    name = "inspect_answer"
    description = """
Use this tool to critique the agent's execution output from a previous step. 
Provide the generated answer and optionally the original question or goal.
This tool will analyze the answer for accuracy, completeness, and reasoning flaws, and suggest improvements if needed."""

    inputs = {
        "generated_answer": {
            "description": "The output or answer produced by the agent.",
            "type": "string",
        },
        "original_question": {
            "description": "[Optional] The original question or task the agent was trying to solve.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    def forward(self, generated_answer: str, original_question: str | None = None) -> str:
        from .smolagents_lib.models import MessageRole

        user_prompt = (
            "Please critique the following agent answer. Identify any flaws, assumptions, missing information, or improvements. "
            "Provide your feedback under these headings:\n"
            "1. Brief Evaluation\n"
            "2. Detailed Critique (with reasoning)\n"
            "3. Suggestions for Improvement"
        )
        if original_question:
            user_prompt += f"\n\nOriginal Question:\n{original_question}"

        user_prompt += f"\n\nGenerated Answer:\n{generated_answer}"

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [{"type": "text", "text": "You are a critical but helpful data science reviewer."}],
            },
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        return self.model(messages).content
