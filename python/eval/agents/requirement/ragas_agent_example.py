import asyncio
import json
import os
import sys
import pandas as pd
from typing import Any, List

# 1. פתרון סופי לבעיית ה-Imports והנתיבים
current_file_path = os.path.dirname(os.path.abspath(__file__)) # .../agents/requirement
eval_root = os.path.abspath(os.path.join(current_file_path, "..", "..")) 

# הוספה לראש הרשימה (index 0) כדי לקבל עדיפות על פני תיקיות מקומיות
if eval_root not in sys.path:
    sys.path.insert(0, eval_root)

# 2. ייבוא מטריקות (מהמיקום החדש למניעת Deprecation)
from ragas import evaluate, EvaluationDataset
from ragas.metrics.collections import AnswerCorrectness, ContextRecall, ExactMatch
from ragas.metrics.base import SingleTurnMetric

# 3. ייבוא התשתית שלנו
from ragasLLM import RagasLLM, create_ragas_evaluation_table
from _utils import print_evaluation_table # עכשיו הוא ימשוך מה-utils הנכון ב-eval
from beeai_framework.backend.chat import ChatModel

# ==========================================
# קונפיגורציה
# ==========================================
NUM_TEST_CASES = 5 # הגדר ל-50 להרצה מלאה
DATASET_PATH = os.path.join(current_file_path, "evaluation_dataset_50_clean.json")

# ==========================================
# 4. מימוש המטריקות (TestCase Metrics)
# ==========================================

class ToolUsageMetric(SingleTurnMetric):
    name: str = "tool_usage"
    async def _ascore(self, row: dict, callbacks: Any = None) -> float:
        expected = row.get("expected_tools", [])
        actual = row.get("tools_called", [])
        if not expected: return 1.0 if not actual else 0.0
        matches = 0
        for exp in expected:
            exp_query = str(exp.get("input_parameters", {}).get("query", "")).lower()
            if any(exp_query in str(act.get("input_parameters", {}).get("query", "")).lower() for act in actual):
                matches += 1
        return float(matches / len(expected))

class FactsSimilarityMetric(SingleTurnMetric):
    name: str = "fact_similarity"
    def __init__(self, llm):
        self.llm = llm
        super().__init__()
    async def _ascore(self, row: dict, callbacks: Any = None) -> float:
        prompt = f"Reference: {row.get('reference')}\nResponse: {row.get('response')}\nScore 1.0 if facts match, 0.5 if partial, 0.0 if not. Return number only."
        try:
            res = await self.llm.generate_text(prompt)
            return float(''.join(c for c in res if c.isdigit() or c=='.'))
        except: return 0.0

# ==========================================
# 5. הרצה
# ==========================================
async def main():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        return

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data[:NUM_TEST_CASES])
    df = df.rename(columns={"input": "user_input", "actual_output": "response", 
                            "expected_output": "reference", "retrieval_context": "retrieved_contexts"})
    
    dataset = EvaluationDataset.from_pandas(df)
    eval_llm = RagasLLM(ChatModel.from_name("ollama:llama3.1:8b"))

    metrics = [
        ExactMatch(), 
        AnswerCorrectness(llm=eval_llm), 
        ToolUsageMetric(), 
        FactsSimilarityMetric(llm=eval_llm), 
        ContextRecall(llm=eval_llm)
    ]

    print(f"🚀 Running Evaluation on {NUM_TEST_CASES} cases...")
    results = evaluate(dataset=dataset, metrics=metrics, llm=eval_llm)

    print("\n" + "="*60 + "\nFINAL RAGAS REPORT\n" + "="*60)
    table = create_ragas_evaluation_table(results, metrics)
    print_evaluation_table(table)

if __name__ == "__main__":
    asyncio.run(main())