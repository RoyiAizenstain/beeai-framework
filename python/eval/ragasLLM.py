# ragasLLM.py
# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
import pandas as pd
from typing import Any, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

from beeai_framework.backend import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware

# Ragas core imports - שימוש ב-EvaluationDataset הרשמי
from ragas import EvaluationDataset
from ragas.llms import BaseRagasLLM

try:
    from _utils import EvaluationRow, EvaluationTable
except ImportError:
    # אם הוא מוצא את הקובץ הלא נכון בתיקיית ה-agents, ננסה לייבא ישירות
    import sys
    import os
    # הגדרת נתיב אבסולוטי לתיקיית eval
    eval_path = os.path.dirname(os.path.abspath(__file__))
    if eval_path not in sys.path:
        sys.path.insert(0, eval_path)
    from _utils import EvaluationRow, EvaluationTable

load_dotenv()

@dataclass
class AgentEvalResult:
    input: str
    expected_output: str
    actual_output: str = ""
    retrieval_context: List[str] = field(default_factory=list)
    tools_called: List[Any] = field(default_factory=list)
    expected_tools: List[Any] = field(default_factory=list)

class RagasLLM(BaseRagasLLM):
    def __init__(self, model: ChatModel):
        self._model = model
        super().__init__()

    async def generate_text(
        self, prompt: str, n: int = 1, temperature: float = 1e-8, 
        stop: Optional[List[str]] = None, callbacks: Any = None,
    ) -> str:
        input_msg = UserMessage(prompt)
        response = await self._model.run([input_msg], temperature=temperature, stream=False).middleware(
            GlobalTrajectoryMiddleware(enabled=os.environ.get("EVAL_LOG_LLM_CALLS", "").lower() == "true")
        )
        return response.get_text_content()

    def get_model_name(self) -> str:
        return f"{self._model.model_id} ({self._model.provider_id})"

async def create_ragas_dataset(*, agent_factory: Any, agent_run: Any, goldens: list) -> EvaluationDataset:
    """
    יוצר EvaluationDataset מתוך ריצות הסוכן בצורה תקינה.
    """
    results = []
    for golden in goldens:
        agent = agent_factory()
        case = AgentEvalResult(
            input=golden.input, 
            expected_output=golden.expected_output, 
            expected_tools=golden.expected_tools
        )
        await agent_run(agent, case)
        
        # בניית השורה בפורמט ש-Ragas מצפה לו
        results.append({
            "user_input": case.input, # Ragas משתמשת ב-user_input בגרסאות החדשות
            "response": case.actual_output,
            "retrieved_contexts": case.retrieval_context if case.retrieval_context else [case.actual_output],
            "reference": case.expected_output,
            "tools_called": [t.model_dump() if hasattr(t, 'model_dump') else t for t in case.tools_called],
            "expected_tools": [t.model_dump() if hasattr(t, 'model_dump') else t for t in case.expected_tools]
        })

    # המרה ל-EvaluationDataset - פונקציה זו מטפלת פנימית ב-Backend
    df = pd.DataFrame(results)
    return EvaluationDataset.from_pandas(df)

def create_ragas_evaluation_table(ragas_result, metrics: List[Any]) -> EvaluationTable:
    metric_names = [getattr(m, 'name', m.__class__.__name__) for m in metrics]
    df = ragas_result.to_pandas()
    
    rows = []
    for idx, row in df.iterrows():
        metric_success_map = {name: bool(row.get(name, 0.0) >= 0.5) for name in metric_names}
        rows.append(EvaluationRow(test_case_label=f"Test case {idx + 1}", results=metric_success_map))
        
    return EvaluationDataset.from_pandas(df)