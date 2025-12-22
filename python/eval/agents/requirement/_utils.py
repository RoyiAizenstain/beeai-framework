# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from typing import Any, List, TypeVar

from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall, Turn
from pydantic import BaseModel

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.types import RequirementAgentRunStateStep
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.tool import Tool
from beeai_framework.utils.strings import to_json


def to_eval_tool_call(step: RequirementAgentRunStateStep, *, reasoning: str | None = None) -> ToolCall:
    if not step.tool:
        raise ValueError("Passed step is missing a tool call.")

    return ToolCall(
        name=step.tool.name,
        description=step.tool.description,
        input_parameters=step.input,
        output=step.output.get_text_content(),
        reasoning=reasoning,
    )


TInput = TypeVar("TInput", bound=BaseModel)


def tool_to_tool_call(
    tool: Tool[TInput, Any, Any], *, input: TInput | None = None, reasoning: str | None = None
) -> ToolCall:
    return ToolCall(
        name=tool.name,
        description=tool.description,
        input_parameters=input.model_dump(mode="json") if input is not None else None,
        reasoning=reasoning,
    )


async def run_agent(agent: RequirementAgent, test_case: LLMTestCase) -> None:
    response = await agent.run(test_case.input)
    test_case.tools_called = []
    test_case.actual_output = response.last_message.text
    state = response.state
    for index, step in enumerate(state.steps):
        if not step.tool:
            continue
        prev_step = state.steps[index - 1] if index > 0 else None
        test_case.tools_called = [
            to_eval_tool_call(
                step,
                reasoning=to_json(prev_step.input, indent=2, sort_keys=False)
                if prev_step and isinstance(prev_step.tool, ThinkTool)
                else None,
            )
            for step in state.steps
            if step.tool and not isinstance(step.tool, FinalAnswerTool)
        ]


def to_conversation_test_case(agent: RequirementAgent, turns: list[Turn]) -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=turns,
        chatbot_role=agent.meta.description or "",
        name="conversation",
        additional_metadata={
            "agent_name": agent.meta.name,
        },
    )


class EvaluationRow(BaseModel):
    """
    Represents a single row in the evaluation table.
    """
    test_case_label: str
    # Map of metric_name -> success (bool)
    results: dict[str, bool]


class EvaluationTable(BaseModel):
    """
    Structured representation of evaluation results for consistent table reporting.
    """
    metric_names: List[str]
    rows: List[EvaluationRow]


def print_evaluation_table(table: EvaluationTable) -> None:
    """
    Prints the evaluation table and adds a success percentage row at the bottom.
    """
    metric_names = table.metric_names
    rows = table.rows

    if not rows:
        print("No test results to display.")
        return

    total_cases = len(rows)
    success_counts = Counter({name: 0 for name in metric_names})
    
    for row in rows:
        for name in metric_names:
            if row.results.get(name, False):
                success_counts[name] += 1

    # Footer with success percentages per metric
    footer = ["Success %"]
    for name in metric_names:
        pct = (success_counts[name] / total_cases * 100) if total_cases else 0
        footer.append(f"{pct:.0f}%")

    # Header
    header = ["Test case"] + metric_names
    
    # Data rows formatted for printing
    formatted_rows = []
    for row in rows:
        formatted_row = [row.test_case_label]
        for name in metric_names:
            formatted_row.append("V" if row.results.get(name, False) else "X")
        formatted_rows.append(formatted_row)

    all_rows = [header] + formatted_rows + [footer]
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

    def fmt_row(row_data):
        return " | ".join(str(cell).ljust(w) for cell, w in zip(row_data, col_widths))

    print("\n=== Evaluation Results Table ===")
    print(fmt_row(header))
    print("-+-".join("-" * w for w in col_widths))
    for row_data in formatted_rows:
        print(fmt_row(row_data))
    print("-+-".join("-" * w for w in col_widths))
    print(fmt_row(footer))
    print("=== End Table ===\n")
