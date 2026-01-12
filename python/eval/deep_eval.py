# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, List, TypeVar

from deepeval.key_handler import KEY_FILE_HANDLER, ModelKeyValues
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from pydantic import BaseModel

from ._utils import EvaluationRow, EvaluationTable

from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import UserMessage
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.utils import ModelLike

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

TSchema = TypeVar("TSchema", bound=BaseModel)


load_dotenv()


class DeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model: ChatModel, *args: Any, **kwargs: Any) -> None:
        self._model = model
        super().__init__(model.model_id, *args, **kwargs)

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        return None

    # pyrefly: ignore [bad-override]
    def generate(self, prompt: str, schema: BaseModel | None = None) -> str:
        """
        Synchronous generate for DeepEval metrics that do not support async.
        Note: This may fail if called from an already running event loop.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If we are in an event loop, we can't use asyncio.run.
            # This is a known limitation when combining sync DeepEval metrics with an async environment.
            # Most modern DeepEval metrics use a_measure/a_generate.
            raise RuntimeError(
                "DeepEvalLLM.generate() called from a running event loop. "
                "Please use the async version (a_generate) or ensure the metric supports async_mode=True."
            )

        return asyncio.run(self.a_generate(prompt, schema))

    # pyrefly: ignore [bad-override]
    async def a_generate(self, prompt: str, schema: TSchema | None = None) -> str:
        input_msg = UserMessage(prompt)
        response = await self._model.run(
            [input_msg],
            response_format=schema.model_json_schema(mode="serialization") if schema is not None else None,
            stream=False,
            temperature=0,
        ).middleware(
            GlobalTrajectoryMiddleware(
                pretty=True, exclude_none=True, enabled=os.environ.get("EVAL_LOG_LLM_CALLS", "").lower() == "true"
            )
        )
        text = response.get_text_content()
        return schema.model_validate_json(text) if schema else text  # type: ignore

    # pyrefly: ignore [bad-override]
    def get_model_name(self) -> str:
        return f"{self._model.model_id} ({self._model.provider_id})"

    @staticmethod
    def from_name(
        name: str | ProviderName | None = None, options: ModelLike[ChatModelParameters] | None = None, **kwargs: Any
    ) -> "DeepEvalLLM":
        name = name or KEY_FILE_HANDLER.fetch_data(ModelKeyValues.LOCAL_MODEL_NAME)
        # pyrefly: ignore [bad-argument-type]
        model = ChatModel.from_name(name, options, **kwargs)
        return DeepEvalLLM(model)


def create_evaluation_table(eval_results, metrics: List[BaseMetric]) -> EvaluationTable:
    """
    Converts DeepEval results into a structured EvaluationTable.
    """
    def _metric_name(metric_obj):
        return getattr(metric_obj, "__name__", None) or metric_obj.__class__.__name__

    metric_names = [_metric_name(m) for m in metrics]

    per_test_results = (
        getattr(eval_results, "results", None)
        or getattr(eval_results, "test_results", None)
        or []
    )

    if isinstance(eval_results, list):
        per_test_results = eval_results

    rows = []
    for idx, test_res in enumerate(per_test_results):
        metrics_data = (
            getattr(test_res, "metrics_data", None)
            or getattr(test_res, "metrics_results", None)
            or []
        )
        
        # Build map for this specific row
        metric_success_map = {}
        for md in metrics_data:
            md_name = (
                getattr(md, "metric_name", None)
                or getattr(md, "name", None)
                or getattr(md, "__name__", None)
                or md.__class__.__name__
            )
            metric_success_map[md_name] = getattr(md, "success", False)

        # Create structured row
        row = EvaluationRow(
            test_case_label=f"Test case {idx + 1}",
            results={name: metric_success_map.get(name, False) for name in metric_names}
        )
        rows.append(row)
        
    return EvaluationTable(metric_names=metric_names, rows=rows)


def print_detailed_report(eval_results) -> None:
    """
    Prints a detailed report for each test case, including inputs, 
    actual outputs, and the specific reasoning for each metric score.
    """
    console = Console()
    
    # Handle different possible formats of eval_results
    results = getattr(eval_results, "test_results", []) or getattr(eval_results, "results", [])
    if isinstance(eval_results, list):
        results = eval_results

    if not results:
        console.print("[bold red]No execution results found to display.[/bold red]")
        return

    console.print("\n[bold cyan]🔍 Detailed Test Case Execution Report[/bold cyan]\n")

    for i, res in enumerate(results):
        # 1. המרת נתוני הקלט והפלט לטבלה נקייה
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_row("[bold yellow]Input:[/bold yellow]", str(res.input))
        
        # הדגשת מצב שבו הפלט ריק (הבעיה שהייתה לך קודם)
        actual_out = str(res.actual_output).strip()
        display_out = f"[white]{actual_out}[/white]" if actual_out else "[bold italic red]EMPTY OUTPUT[/bold italic red]"
        
        info_table.add_row("[bold green]Actual Output:[/bold green]", display_out)
        info_table.add_row("[bold blue]Expected Output:[/bold blue]", str(res.expected_output))

        # 2. בניית טבלת המטריקות והנימוקים
        metrics_table = Table(show_header=True, header_style="bold magenta", box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", justify="center")
        metrics_table.add_column("Status", justify="center")
        metrics_table.add_column("Reasoning (LLM Judge)", style="dim", width=60)

        for md in (res.metrics_data or []):
            status = "[green]PASS[/green]" if md.success else "[red]FAIL[/red]"
            metrics_table.add_row(
                str(md.name),
                f"{md.score:.2f}",
                status,
                str(md.reason or "No explanation provided.")
            )

        # 3. הדפסה בתוך פאנל מופרד לכל מקרה בדיקה
        console.print(
            Panel(
                Group(info_table, "\n[bold underline]Metrics Breakdown:[/bold underline]", metrics_table),
                title=f"[bold white]Test Case #{i+1}[/bold white]",
                border_style="bright_blue",
                padding=(1, 2)
            )
        )
    
