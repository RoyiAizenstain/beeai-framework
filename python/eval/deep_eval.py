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
        raise NotImplementedError()

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

    
    
