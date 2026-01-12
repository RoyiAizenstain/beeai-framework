import asyncio
import sys
import os
import pandas as pd

# הגדרת נתיבים
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ragasLLM import create_ragas_dataset
from agents.requirement._utils import run_agent
from deepeval.dataset import Golden

# הגדרת אובייקט בדיקה
debug_golden = Golden(
    input="Who directed the 2014 film Fury?",
    expected_output="David Ayer",
    expected_tools=[{"name": "Wikipedia"}]
)

def agent_factory():
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.backend.chat import ChatModel
    from beeai_framework.tools.search.wikipedia import WikipediaTool
    return RequirementAgent(llm=ChatModel.from_name("ollama:llama3.1:8b"), tools=[WikipediaTool()])

async def debug():
    print("--- Starting Dataset Debug ---")
    try:
        # יצירת ה-Dataset
        dataset = await create_ragas_dataset(
            agent_factory=agent_factory, 
            agent_run=run_agent, 
            goldens=[debug_golden]
        )
        
        # הדפסת מידע לאימות
        print(f"\n✅ Dataset Created! Type: {type(dataset)}")
        df = dataset.to_pandas()
        print("\nColumns found:", df.columns.tolist())
        print("Sample response:", df['response'].iloc[0])
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug())