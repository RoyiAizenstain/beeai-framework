# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from pydantic import BaseModel

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.backend import ChatModel
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.types import ChatModelInput, ChatModelOutput
from beeai_framework.context import RunContext
from beeai_framework.evaluation.adapters.ragas_llm import InstructorRagasLLM


class _AnswerSchema(BaseModel):
    answer: str


class _StructuredDummyModel(ChatModel):
    """Dummy model that returns a fixed structured output, mimicking a well-behaved backend."""

    model_id = "structured_dummy_model"
    # pyrefly: ignore [bad-override]
    provider_id = "ollama"

    def __init__(self, output_structured: Any) -> None:
        super().__init__()
        self._output_structured = output_structured

    # pyrefly: ignore [bad-param-name-override]
    async def _create(self, input: ChatModelInput, _: RunContext) -> ChatModelOutput:
        return ChatModelOutput(output=[], output_structured=self._output_structured)

    # pyrefly: ignore [bad-param-name-override]
    async def _create_stream(self, input: ChatModelInput, _: RunContext) -> AsyncGenerator[ChatModelOutput]:
        yield ChatModelOutput(output=[], output_structured=self._output_structured)


@pytest.mark.unit
def test_instructor_ragas_llm_from_name_forwards_kwargs() -> None:
    llm = InstructorRagasLLM.from_name("ollama:llama3.1", base_url="http://somewhere:12345")

    assert isinstance(llm.model, OllamaChatModel)
    assert llm.model._settings["base_url"] == "http://somewhere:12345/v1"


@pytest.mark.unit
def test_instructor_ragas_llm_from_name_without_kwargs() -> None:
    llm = InstructorRagasLLM.from_name("ollama:llama3.1")

    assert isinstance(llm.model, OllamaChatModel)


@pytest.mark.unit
def test_instructor_ragas_llm_wraps_existing_chat_model() -> None:
    model = ChatModel.from_name("ollama:llama3.1")

    llm = InstructorRagasLLM(model)

    assert llm.model is model


@pytest.mark.asyncio
@pytest.mark.unit
async def test_agenerate_returns_the_backends_structured_output() -> None:
    expected = _AnswerSchema(answer="Delhi")
    model = _StructuredDummyModel(output_structured=expected)
    llm = InstructorRagasLLM(model)

    result = await llm.agenerate("What city?", _AnswerSchema)

    assert result is expected


@pytest.mark.asyncio
@pytest.mark.unit
async def test_agenerate_raises_when_backend_fails_to_produce_structured_output() -> None:
    model = _StructuredDummyModel(output_structured=None)
    llm = InstructorRagasLLM(model)

    with pytest.raises(ChatModelError):
        await llm.agenerate("What city?", _AnswerSchema)
