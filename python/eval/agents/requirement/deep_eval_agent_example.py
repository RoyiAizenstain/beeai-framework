# --- Standard Library Imports ---
import asyncio
import json
import logging
import os
import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from collections import Counter
from typing import Any, List

# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()



# --- Configuration & Logging ---
def setup_logger():
    """Sets up a logger that outputs to both a file and the console."""
    logger = logging.getLogger("DeepEvalAgentExample")
    logger.setLevel(logging.INFO)
    
    log_file = Path("evaluation.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

# --- Path Configuration ---
# Ensure the monorepo's python package root is importable
CURRENT_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = CURRENT_DIR.parent.parent.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

# Ensure current directory is in path for local imports
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# --- Third-Party & Framework Imports ---
import pytest
from deepeval import evaluate
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import (
    BaseMetric,
    ContextualRecallMetric,
    ExactMatchMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel, ToolMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.tools.code import PythonTool, LocalPythonStorage
from beeai_framework.tools.think import ThinkTool

# --- Local Project Imports ---
from eval.deep_eval import (
    DeepEvalLLM,
    create_evaluation_table,
    print_detailed_report,
)
from eval._utils import (
    print_evaluation_table,
    run_agent_with_fail_safe,
)

test_cases_num = 1

# --- DeepEval Custom Metrics ---

class FactsSimilarityMetric(BaseMetric):
    """
    Evaluates how many expected facts are covered in the retrieved context using an LLM judge.
    """
    success: bool = False

    def __init__(self, model: DeepEvalLLM | None = None, threshold: float = 0.5):
        super().__init__()
        self.model: DeepEvalLLM = model or DeepEvalLLM.from_name("ollama:llama3.1:8b")
        self.threshold = threshold
        self.async_mode = True

    def _get_expected(self, test_case: LLMTestCase) -> list[str]:
        if hasattr(test_case, "expected_facts"):
            return getattr(test_case, "expected_facts")
        metadata = getattr(test_case, "additional_metadata", None) or {}
        return metadata.get("expected_facts", [])

    async def a_measure(self, test_case: LLMTestCase) -> float:
        actual_facts = getattr(test_case, "retrieval_context", [])
        expected_facts = self._get_expected(test_case)

        if not expected_facts:
            return 1.0 if not actual_facts else 0.0
        if not actual_facts:
            self.score = 0.0
            return 0.0

        prompt = (
            "Role: Expert Information Auditor\n"
            "Task: Evaluate the coverage of 'Expected Facts' within the 'Retrieved Context'.\n\n"
            f"Expected Facts (Ground Truth):\n{expected_facts}\n\n"
            f"Retrieved Context (Agent Output):\n{actual_facts}\n\n"
            "Instructions:\n"
            "1. Break down the Expected Facts into core independent claims.\n"
            "2. For each claim, check if it is supported by the Retrieved Context.\n"
            "3. Calculation: (Number of supported claims) / (Total number of expected claims).\n\n"
            "Final Score: Output ONLY the numerical score between 0.0 and 1.0."
        )

        text = await self.model.a_generate(prompt)
        
        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
        score = float(numbers[0]) if numbers else 0.0

        self.score = max(0.0, min(1.0, score))
        self.success = self.score >= self.threshold
        return self.score

    def measure(self, test_case: LLMTestCase) -> float:
        raise NotImplementedError("Use a_measure() instead.")

    def is_successful(self) -> bool:
        return getattr(self, "success", False)

    @property
    def __name__(self):
        return "FactsSimilarityMetric"

class AnswerLLMJudgeMetric(BaseMetric):
    """
    Uses an LLM as a judge to determine if the Model Answer is semantically identical to the Expected Answer.
    """
    success: bool = False

    def __init__(self, model: DeepEvalLLM | None = None, threshold: float = 0.5):
        super().__init__()
        self.model: DeepEvalLLM = model or DeepEvalLLM.from_name("ollama:llama3.1:8b")
        self.threshold = threshold
        self.async_mode = True

    async def a_measure(self, test_case: LLMTestCase) -> float:
        actual = (test_case.actual_output or "").strip()
        expected = (test_case.expected_output or "").strip()

        if not expected:
            return 1.0 if not actual else 0.0

        prompt = (
            "You are an expert evaluator. Your goal is to determine if the Model Answer is semantically identical to the Expected Answer.\n\n"
            f"Question: {test_case.input}\n"
            f"Expected Answer: {expected}\n"
            f"Model Answer: {actual}\n\n"
            "Evaluation Criteria:\n"
            "1. If the answers share the same core meaning (e.g., 'Messi' vs 'Lionel Messi'), give 1.0.\n"
            "2. If the answer is partially correct but missing key info, give 0.5.\n"
            "3. If the answer is wrong or contradicts the expected, give 0.0.\n\n"
            "Instructions: Provide your reasoning in one sentence, then on a new line provide the score as: 'Score: <number>'"
        )

        text = await self.model.a_generate(prompt)
        
        try:
            import re
            match = re.search(r"Score:\s*([\d\.]+)", text)
            if match:
                score = float(match.group(1))
            else:
                score = float(str(text).strip())
        except:
            score = 0.0

        self.score = max(0.0, min(1.0, score))
        self.success = self.score >= self.threshold
        return self.score

    def measure(self, test_case: LLMTestCase) -> float:
        raise NotImplementedError("Use a_measure() instead.")

    def is_successful(self) -> bool:
        return getattr(self, "success", False)

    @property
    def __name__(self) -> str:
        return "AnswerLLMJudgeMetric"

class ToolUsageMetric(BaseMetric):
    """
    Compares the tools called by the agent with the expected tools.
    """
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.score = 0.0
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        expected_tools = getattr(test_case, "expected_tools", []) or \
                         (test_case.additional_metadata.get("expected_tools_detail") if test_case.additional_metadata else [])
        
        actual_tools = getattr(test_case, "tools_called", []) or \
                       (test_case.additional_metadata.get("actual_tools_detail") if test_case.additional_metadata else [])
        
        if not expected_tools:
            self.score = 1.0 if not actual_tools else 0.0
            self.success = self.score >= self.threshold
            return self.score

        matches = 0
        used_actual_indices = set()

        for expected in expected_tools:
            exp_name = expected.name
            exp_query = str(expected.input_parameters.get("query", "")).lower()
            
            for i, actual in enumerate(actual_tools):
                if i in used_actual_indices:
                    continue
                
                act_name = actual.name
                act_query = str(actual.input_parameters.get("query", "")).lower()
                
                if exp_name == act_name and exp_query in act_query:
                    matches += 1
                    used_actual_indices.add(i)
                    break
        
        self.score = matches / len(expected_tools)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "ToolUsageMetric"

# --- Utility Functions ---

def create_calculator_tool() -> Tool:
    """Creates a PythonTool configured for mathematical calculations."""
    storage = LocalPythonStorage(
        local_working_dir=tempfile.mkdtemp("code_interpreter_source"),
        interpreter_working_dir=os.getenv("CODE_INTERPRETER_TMPDIR", "./tmp/code_interpreter_target"),
    )

    python_tool = PythonTool(
        code_interpreter_url=os.getenv("CODE_INTERPRETER_URL", "http://127.0.0.1:50081"),
        storage=storage,
    )
    return python_tool

def extract_retrieval_context(messages) -> List[str]:
    """Extracts document descriptions from VectorStoreSearch tool messages."""
    retrieval_context = []
    
    for message in messages:
        if isinstance(message, ToolMessage) and message.content and len(message.content) > 0:
            if hasattr(message.content[0], 'tool_name') and message.content[0].tool_name == "VectorStoreSearch":
                try:
                    for content_item in message.content:
                        if hasattr(content_item, 'result') and content_item.result:
                            result_data = json.loads(content_item.result) if isinstance(content_item.result, str) else content_item.result
                            if isinstance(result_data, list):
                                for doc in result_data:
                                    if isinstance(doc, dict) and 'description' in doc:
                                        retrieval_context.append(doc['description'])
                except Exception as e:
                    print(f"Warning: Failed to parse retrieval context: {e}")
                    continue
    return retrieval_context

# --- Agent Factory ---

async def create_agent() -> RequirementAgent:
    """Instantiates a RequirementAgent with pre-configured tools and instructions."""
    wiki_tool = WikipediaTool() 
    calculator_tool = create_calculator_tool()

    model_name = os.environ.get("AGENT_CHAT_MODEL_NAME", os.environ.get("EVAL_CHAT_MODEL_NAME", "ollama:llama3.1:8b"))

    llm = ChatModel.from_name(
        model_name,
        {"allow_parallel_tool_calls": True},
    )

    JSON_SCHEMA_STRING = """{
        "answer": "<concise, specific answer only (e.g., 'Delhi')>",
        "supporting_sentences": ["<sentence 1>", "<sentence 2>"]
    }"""
    
    agent = RequirementAgent(
        llm=llm, 
        tools=[wiki_tool, OpenMeteoTool(), calculator_tool, ThinkTool()],
        memory=UnconstrainedMemory(),
        role="You are an expert Multi-hop Question Answering (QA) agent. Your primary role is to query the available data sources, extract relevant information and combine information from the provided context to answer the user's question. Before searching, use the ThinkTool to plan your search strategy. Answer in JSON format only.",
        instructions=[
            "RULES and CONSTRAINTS:",
            "1. SOURCE ADHERENCE: Your final answer MUST be based ONLY on the retrieved context. If you cannot find the answer after multiple search attempts, state clearly what information is missing.",
            
            "2. SEARCH STRATEGY: Wikipedia works best with entities (names, places, events). "
            "If your search returns 'No results', DECOMPOSE the question and search for the main subjects separately. "
            "Example: Instead of 'widow affected by X decision', search for 'X decision' first.",
            
            "3. MULTI-HOP: You must perform as many steps as needed. If one tool call doesn't give the full answer, use the information gained to make a better second tool call.",
            
            "4. OUTPUT FORMAT: You must ALWAYS respond in the required JSON format. Never return an empty 'response' field if you found any partial information.",
            
            "5. THE RESPONSE JSON SCHEMA: " + JSON_SCHEMA_STRING
        ],
    )
    return agent

# --- Execution Phases ---

async def run_agent_batch_execution(test_data):
    """Phase 1: Batch Execution using fail-safe utility."""
    async def agent_run_helper(agent, question):
        logger.info(f"Running agent for question: {question[:50]}...")
        response = await agent.run(question)
        return {
        "text": response.last_message.text,
        "memory": [m.to_json_safe() for m in response.state.memory.messages] # שמירת ההיסטוריה כ-JSON
    }

    checkpoint_path = CURRENT_DIR / "agent_run_checkpoint.pkl"
    return await run_agent_with_fail_safe(
        inputs=[item["question"] for item in test_data],
        agent_factory=create_agent,
        run_fn=agent_run_helper,
        temp_file=checkpoint_path,
        reinstantiate=True,
        max_retries=3
    )

def create_test_cases_from_responses(test_data, agent_responses):
    """Phase 2: Transformation of agent responses into LLMTestCase objects."""
    test_cases = []
    for i, (item, response) in enumerate(zip(test_data, agent_responses)):
        if response is None:
            logger.warning(f"Skipping test case {i+1} due to execution failure.")
            continue
            
        question = item["question"]
        HotpotQA_expected_output = item["answer"]
        HotpotQA_context = item["relevant_sentences"]
        HotpotQA_expected_tools = {"Wikipedia": item["wiki_times"]}
        supporting_titles = item["supporting_titles"]
        
        HotpotQA_tools_used_detail = [ToolCall(name="Wikipedia", input_parameters={'query': name}) for name in supporting_titles]

        actual_output = response["text"] if isinstance(response, dict) else response.last_message.text
        memory = response["memory"] if isinstance(response, dict) else response.state.memory.messages
        agent_tools_list = []
        agent_supporting_sentences = []

        # Process message history for tool calls and facts
        for msg in memory:
            msg_data = msg if isinstance(msg, dict) else msg.to_json_safe()
            role = msg_data.get("role")
            content_list = msg_data.get("content", [])

            if role == "assistant":
                for content_item in content_list:
                    if content_item.get("type") == "tool-call" and content_item.get("tool_name") != "final_answer":
                        tool_name = content_item.get("tool_name")
                        raw_args = content_item.get("args", "{}")
                        try:
                            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except:
                            parsed_args = {"query": str(raw_args)}
                        agent_tools_list.append(ToolCall(name=tool_name, input_parameters=parsed_args))

            elif role == "tool":
                for content_item in content_list:
                    raw_result = content_item.get("result") or content_item.get("text", "")
                    try:
                        data = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                        fact_text = data[0].get('description', str(data[0])) if isinstance(data, list) and data else str(data)
                    except:
                        fact_text = str(raw_result)

                    clean_fact = fact_text.strip()
                    if clean_fact and "no results" not in clean_fact.lower() and len(clean_fact) > 20:
                        if clean_fact not in agent_supporting_sentences:
                            agent_supporting_sentences.append(clean_fact[:500])

        agent_tool_usage_dict = dict(Counter([tc.name for tc in agent_tools_list]))

        # Parse agent output JSON
        try:
            loaded_data = json.loads(actual_output)
            agent_response_json = loaded_data if isinstance(loaded_data, dict) else {}
        except:
            agent_response_json = {}

        agent_final_answer = agent_response_json.get("answer") or agent_response_json.get("final_answer") or actual_output
        
        supporting_sentences_from_agent = agent_response_json.get("supporting_sentences", None)
        agent_supporting_sentences = supporting_sentences_from_agent if isinstance(supporting_sentences_from_agent, list) else agent_supporting_sentences

        test_case = LLMTestCase(
            input=question,
            actual_output=agent_final_answer,
            expected_output=HotpotQA_expected_output,
            retrieval_context=agent_supporting_sentences,
            context=HotpotQA_context,
            tools_called=agent_tools_list,
            expected_tools=HotpotQA_tools_used_detail,
            additional_metadata={
                "expected_facts": HotpotQA_context,
                "tool_usage": agent_tool_usage_dict,
                "expected_tool_usage": HotpotQA_expected_tools,
                "supporting_titles": supporting_titles,
            }
        )

        print(f"----- TEST CASE {i+1} -----")
        print(f"Question: {question}")
        print(f"Expected Answer: {HotpotQA_expected_output}")
        print(f"Actual Answer: {agent_final_answer}")
        print(f"Expected Facts (Ground Truth): {HotpotQA_context}")
        print(f"Retrieved Facts by Agent: {agent_supporting_sentences}")
        print(f"Expected Tools Usage: {HotpotQA_expected_tools}")
        print(f"Actual Tools Called: {agent_tool_usage_dict}")
        print("---------------------")
        test_cases.append(test_case)

    return test_cases

async def agent_run(num_rows: int = 50):
    """Orchestrates Phase 1 (Execution) and Phase 2 (Transformation)."""
    dataset_path = Path(__file__).parent / "evaluation_dataset_50_clean.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_data = test_data[:min(num_rows, 50)]
    agent_responses = await run_agent_batch_execution(test_data)
    return create_test_cases_from_responses(test_data, agent_responses)

# --- Evaluation Phase ---

async def run_evaluation(test_cases: List[LLMTestCase], metrics: List[BaseMetric]):
    """Runs DeepEval evaluation loop with incremental result saving and RESUME logic."""
    pkl_path = Path(__file__).parent / "eval_results_raw.pkl"
    all_test_results = []
    
    # Check for existing evaluation results to resume
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                all_test_results = pickle.load(f)
            logger.info(f"Loaded {len(all_test_results)} existing evaluation results. Resuming...")
        except Exception as e:
            logger.error(f"Failed to load existing evaluation results: {e}")
            all_test_results = []

    start_idx = len(all_test_results)

    # Evaluate only the remaining test cases
    for i in range(start_idx, len(test_cases)):
        test_case = test_cases[i]
        logger.info(f"Evaluating test case {i+1}/{len(test_cases)}...")
        try:
            res = evaluate(
                test_cases=[test_case], 
                metrics=metrics,
                display_config=DisplayConfig(show_indicator=False, print_results=False, verbose_mode=False)
            )
            
            step_results = getattr(res, "results", None) or getattr(res, "test_results", None) or []
            for result in step_results:
                print(f"\n--- METRIC SCORES FOR TEST CASE {i} ---")
                for md in result.metrics_data:
                    status = "✅" if md.success else "❌"
                    print(f"{status} {md.name}: {md.score:.2f}" + (f" (Reason: {md.reason})" if md.reason else ""))
                print("---------------------------------------\n")
            all_test_results.extend(step_results)
            
        except Exception as eval_exc:
            logger.error(f"Error evaluating test case {i+1}: {eval_exc}")
            traceback.print_exc()
        finally:
            if all_test_results:
                try:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(all_test_results, f)
                    logger.info(f"Progress saved to {pkl_path}")
                except Exception as p_err:
                    logger.error(f"Critical: Could not write PKL file: {p_err}")

    return all_test_results

# --- Main Entry Point ---

@pytest.mark.asyncio
async def test_rag() -> None:
    """End-to-end evaluation flow for the RAG agent."""
    global test_cases_num
    test_cases = await agent_run(test_cases_num)

    eval_model_name = os.environ.get("EVAL_CHAT_MODEL_NAME", "ollama:llama3.1:8b")
    os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1000")
    eval_model = DeepEvalLLM.from_name(eval_model_name)

    metrics = [
        ExactMatchMetric(threshold=1.0),
        AnswerLLMJudgeMetric(model=eval_model, threshold=0.7),
        ToolUsageMetric(),
        FactsSimilarityMetric(model=eval_model),
        ContextualRecallMetric(model=eval_model, threshold=0.7),
    ]

    all_test_results = await run_evaluation(test_cases, metrics)
    table = create_evaluation_table(all_test_results, metrics)
    print_evaluation_table(table)
    print_detailed_report(all_test_results)

if __name__ == "__main__":
    asyncio.run(test_rag())


