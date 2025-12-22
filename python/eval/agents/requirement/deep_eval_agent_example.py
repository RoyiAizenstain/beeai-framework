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
from typing import List

# --- Logger Configuration ---
def setup_logger():
    logger = logging.getLogger("DeepEvalAgentExample")
    logger.setLevel(logging.INFO)
    
    # Create file handler
    log_file = Path("evaluation.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()


# --- Path Configuration (Must run before local imports) ---
# Ensure the monorepo's python package root is importable
CURRENT_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = CURRENT_DIR.parent.parent.parent # points to .../python
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

# Ensure current directory is in path for local imports like ToolUsageMetric, _utils
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# --- Third-Party Library Imports ---
import pytest
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ArgumentCorrectnessMetric,
    BaseMetric,
    ContextualRecallMetric,
    ExactMatchMetric,
    FaithfulnessMetric,
    GEval,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

# --- Framework Specific Imports (beeai-framework) ---
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel, ToolMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.tools.code import PythonTool, LocalPythonStorage
from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.errors import FrameworkError

# --- Local Project Imports ---
from eval.deep_eval import (
    DeepEvalLLM,
    create_evaluation_table,
)
from eval._utils import (
    EvaluationRow,
    EvaluationTable,
    print_evaluation_table,
)


# --- Environment Setup ---
load_dotenv()

class FactsSimilarityMetric(BaseMetric):
    # Default so DeepEval's MetricData.success sees a proper boolean
    success: bool = False

    def __init__(self, model: DeepEvalLLM | None = None, threshold: float = 0.5):
        super().__init__()
        # DeepEval expects model to be a DeepEvalBaseLLM; we use our wrapper.
        self.model: DeepEvalLLM = model or DeepEvalLLM.from_name("ollama:llama3.1:8b")
        self.threshold = threshold
        # Let DeepEval use async execution path (a_measure)
        self.async_mode = True

    def _get_expected(self, test_case: LLMTestCase) -> list[str]:
        if hasattr(test_case, "expected_facts"):
            return getattr(test_case, "expected_facts")
        metadata = getattr(test_case, "additional_metadata", None) or {}
        return metadata.get("expected_facts", [])

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        """Async entrypoint DeepEval actually uses; we ignore the extra flags."""
        actual_facts = getattr(test_case, "retrieval_context", [])
        expected_facts = self._get_expected(test_case)
        if not expected_facts:
            score = 1.0 if not actual_facts else 0.0
            self.score = score
            self.success = score >= self.threshold
            return score

        prompt = (
            "You are an evaluator.\n"
            "Compare the two lists of supporting facts.\n\n"
            f"Actual facts:\n{actual_facts}\n\n"
            f"Expected facts:\n{expected_facts}\n\n"
            "Return ONLY a number between 0 and 1 (no text), where:\n"
            "0 = completely different, 1 = identical in meaning.\n"
        )
        text = await self.model.a_generate(prompt)  # uses DeepEvalLLM.a_generate
        score = float(str(text).strip())
        score = max(0.0, min(1.0, score))
        self.score = score
        self.success = score >= self.threshold
        return score

    def measure(self, test_case: LLMTestCase) -> float:
        """Synchronous wrapper for environments that call measure() instead of a_measure()."""
        import asyncio

        return asyncio.run(self.a_measure(test_case))

    def is_successful(self) -> bool:
        return getattr(self, "success", False)

    @property
    def __name__(self):
        return "FactsSimilarityMetric"

class AnswerLLMJudgeMetric(BaseMetric):
    """
    Uses an LLM as a judge to compare the actual answer vs the expected answer.
    Returns a semantic similarity score between 0 and 1.
    """

    success: bool = False  # ensure MetricData.success is always a bool

    def __init__(self, model: DeepEvalLLM | None = None, threshold: float = 0.5):
        super().__init__()
        # DeepEval expects model to be a DeepEvalBaseLLM; we use our wrapper.
        self.model: DeepEvalLLM = model or DeepEvalLLM.from_name("ollama:llama3.1:8b")
        self.threshold = threshold
        self.async_mode = True  # DeepEval will call a_measure

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        actual = (test_case.actual_output or "").strip()
        expected = (test_case.expected_output or "").strip()

        # If no expected answer, treat as trivial pass/fail
        if not expected:
            score = 1.0 if not actual else 0.0
            self.score = score
            self.success = score >= self.threshold
            return score

        prompt = (
            "You are an evaluator.\n"
            "Compare the model's answer to the expected answer.\n\n"
            f"Question:\n{test_case.input}\n\n"
            f"Model answer:\n{actual}\n\n"
            f"Expected answer:\n{expected}\n\n"
            "Return ONLY a number between 0 and 1 (no text), where:\n"
            "0 = completely incorrect or unrelated,\n"
            "1 = fully correct and equivalent in meaning.\n"
        )

        text = await self.model.a_generate(prompt)  # DeepEvalLLM async call
        try:
            score = float(str(text).strip())
        except Exception:
            score = 0.0

        score = max(0.0, min(1.0, score))
        self.score = score
        self.success = score >= self.threshold
        return score

    def measure(self, test_case: LLMTestCase) -> float:
        """Sync wrapper in case something calls measure() directly."""
        import asyncio
        return asyncio.run(self.a_measure(test_case))

    def is_successful(self) -> bool:
        return getattr(self, "success", False)

    @property
    def __name__(self) -> str:
        return "AnswerLLMJudgeMetric"

class ToolUsageMetric(BaseMetric):
    """
    Compares actual tool usage against expected tool usage.
    expected_tool_usage בדוגמה:
        {"Wikipedia": 2, "PythonTool": 1}
    """

    # Default so DeepEval's MetricData.success sees a proper boolean
    success: bool = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.async_mode = False  # חובה כי אין measure אסינכרוני

    def _get_tool_usage(self, test_case: LLMTestCase, key: str) -> dict[str, int]:
        if hasattr(test_case, key):
            return getattr(test_case, key) or {}
        metadata = getattr(test_case, "additional_metadata", None) or {}
        return metadata.get(key, {}) or {}

    def measure(self, test_case: LLMTestCase) -> float:
        actual_tool_usage = self._get_tool_usage(test_case, "tool_usage")
        expected_tool_usage = self._get_tool_usage(test_case, "expected_tool_usage")

        all_tools = set(actual_tool_usage.keys()) | set(expected_tool_usage.keys())
        if not all_tools:
            score = 1.0
            self.score = score
            self.success = score >= self.threshold
            return score

        # =====================
        # חלק 1: השוואת כלים קיימים (0.75)
        # =====================
        matching_tools = sum(
            1 for tool in all_tools if tool in actual_tool_usage and tool in expected_tool_usage
        )
        existence_score = 0.75 * (matching_tools / len(all_tools))

        # =====================
        # חלק 2: השוואת כמויות שימוש בכל כלי (0.25)
        # =====================
        count_score_per_tool = []
        for tool in expected_tool_usage:
            actual_count = actual_tool_usage.get(tool, 0)
            expected_count = expected_tool_usage[tool]
            if actual_count == expected_count:
                count_score_per_tool.append(1.0)
            else:
                count_score_per_tool.append(0.0)
        if count_score_per_tool:
            count_score = 0.25 * (sum(count_score_per_tool) / len(count_score_per_tool))
        else:
            count_score = 0.25  # אין כלים צפוים → נותנים את הציון המלא לחלק זה

        # =====================
        # ציון סופי
        # =====================
        final_score = existence_score + count_score
        self.score = final_score
        self.success = final_score >= self.threshold
        return final_score


    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return getattr(self, "success", False)

    @property
    def __name__(self):
        return "ToolUsageMetric"




def count_tool_usage(messages):
    tool_counter = Counter()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            for item in msg.content:
                tool_name = getattr(item, "tool_name", None)
                if tool_name and tool_name != "final_answer":
                    tool_counter[tool_name] += 1

    return dict(tool_counter)

def create_calculator_tool() -> Tool:
    """
    Create a PythonTool configured for mathematical calculations.
    """
    storage = LocalPythonStorage(
        local_working_dir=tempfile.mkdtemp("code_interpreter_source"),
        # CODE_INTERPRETER_TMPDIR should point to where code interpreter stores it's files
        interpreter_working_dir=os.getenv("CODE_INTERPRETER_TMPDIR", "./tmp/code_interpreter_target"),
    )

    python_tool = PythonTool(
        code_interpreter_url=os.getenv("CODE_INTERPRETER_URL", "http://127.0.0.1:50081"),
        storage=storage,
    )
    return python_tool

test_cases_num = 1


async def create_agent() -> RequirementAgent:
    """
    Create a RequirementAgent with RAG and Wikipedia capabilities.
    """
    #vector_store = await setup_vector_store()
    #need it?
    vector_store = True
    if vector_store is None:
        raise FileNotFoundError(
            "Failed to instantiate Vector Store. "
            "Either set POPULATE_VECTOR_DB=True in your .env file, or ensure the database file exists."
        )
    search_tool = VectorStoreSearchTool(vector_store=vector_store)

    wiki_tool = WikipediaTool() 
    calculator_tool = create_calculator_tool()

    # Use local Ollama without relying on environment variables
    # Allow overriding the agent model; default aligns with eval model naming
    model_name = os.environ.get("AGENT_CHAT_MODEL_NAME", os.environ.get("EVAL_CHAT_MODEL_NAME", "ollama:llama3.1:8b"))
    llm = ChatModel.from_name(
        model_name,
        {"allow_parallel_tool_calls": True},
    )

    # Create RequirementAgent with multiple tools
    # tools: WikipediaTool for general knowledge, PythonTool for calculations, OpenMeteoTool for weather data

    #Format in Jason:
    #Final answer 
    #List of supporting sentences
    #explanation of reasoning for each sentence by its number
    #tool that was used
    #
    JSON_SCHEMA_STRING = """{
        "answer": "<concise, specific answer only (e.g., 'Delhi')>",
        "tool_used": [{"tool": "...", "times_used": 1}],
        "supporting_titles": ["<title 1>", "<title 2>"],
        "supporting_sentences": ["<sentence 1>", "<sentence 2>"],
        "reasoning_explanation": [{"step": 1, "logic": "The reasoning step"}]
    }"""
    
    agent = RequirementAgent(
        llm=llm, 
        tools=[wiki_tool,OpenMeteoTool(), calculator_tool],
        memory=UnconstrainedMemory(),
        role="You are an expert Multi-hop Question Answering (QA) agent. Your primary role is to extract and combine information from the provided context to answer the user's question. Answer in jason format only.",
        instructions=[
            "RULES and CONSTRAINTS:",
            "1. SOURCE ADHERENCE (NO HALLUCINATION): Your final answer MUST be based ONLY on the context you retrieve from the provided tools (VectorStoreSearchTool or WikipediaTool). Do not use external knowledge.",
            "2. MULTI-HOP: You must perform multi-step reasoning or use multiple tools/retrievals if the question requires it.",
            "3. FINAL FORMAT: Your ONLY final output MUST be a single, valid JSON object adhering strictly to the required keys: answer, tool_used, supporting_titles, supporting_sentences, reasoning_explanation. The final_answer must be concise and specific (e.g., just 'Delhi', not a full sentence). Do not include any text outside the JSON block.",
            "4. THE JSON SCHEMA STRING: " + JSON_SCHEMA_STRING
        ],

    )
    return agent

def extract_retrieval_context(messages) -> List[str]:
    """
    Extract retrieval context from tool messages in the message history.
    Looks for ToolMessage with VectorStoreSearch tool_name and extracts document descriptions.
    """
    retrieval_context = []
    
    for message in messages:
        if isinstance(message, ToolMessage) and message.content and len(message.content) > 0:
            if hasattr(message.content[0], 'tool_name') and message.content[0].tool_name == "VectorStoreSearch":
                try:
                    # Extract the tool result from the message content
                    for content_item in message.content:
                        if hasattr(content_item, 'result') and content_item.result:
                            # Parse the JSON result
                            result_data = json.loads(content_item.result) if isinstance(content_item.result, str) else content_item.result
                            
                            # Extract descriptions from each document
                            if isinstance(result_data, list):
                                for doc in result_data:
                                    if isinstance(doc, dict) and 'description' in doc:
                                        retrieval_context.append(doc['description'])
                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                    # If parsing fails, skip this message
                    print(f"Warning: Failed to parse retrieval context: {e}")
                    continue
    
    return retrieval_context

async def create_rag_test_cases(num_rows: int = 50):
    """
    Create RAG test cases by directly invoking the agent and extracting retrieval context.
    """
    agent = await create_agent()
    
    test_cases = []

    dataset_path = Path(__file__).parent / "evaluation_dataset_50_clean.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Load only requested number of rows (capped at 50)
    test_data = test_data[:min(num_rows, 50)]

    # Per-question stubbed responses (can diverge from ground truth)
    # Stubbed responses now mirror the dataset entries exactly
    # stub_map = {
    #     "Which magazine was started first Arthur's Magazine or First for Women?": {
    #         "answer": "Arthur's Magazine",
    #         "titles": ["Arthur's Magazine", "First for Women"],
    #         "sentences": [
    #             "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",
    #             "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
    #         ],
    #         "times_used": 2,
    #     },
    #     "The Oberoi family is part of a hotel company that has a head office in what city?": {
    #         "answer": "Delhi",
    #         "titles": ["Oberoi family", "The Oberoi Group"],
    #         "sentences": [
    #             "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.",
    #             "The Oberoi Group is a hotel company with its head office in Delhi.",
    #         ],
    #         "times_used": 2,
    #     },
    # }

    for i, item in enumerate(test_data):
        question = item["question"]
        logger.info(f"Running agent for test case {i+1}/{len(test_data)}: {question[:50]}...")
        
        HotpotQA_expected_output = item["answer"]
        HotpotQA_context = item["relevant_sentences"]
        HotpotQA_expected_tools = {"Wikipedia": item["wiki_times"]}
        supporting_titles = item["supporting_titles"]
        HotpotQA_tools_used = []
        for name in supporting_titles:
            HotpotQA_tools_used.append(ToolCall(name="Wikipedia", input_parameters={'query': name}))

        # Run the agent
        response = await agent.run(question)
        state = response.state
        memory = state.memory.messages
        actual_output = response.last_message.text
        agent_tool_usage_times = count_tool_usage(memory)


        # print(actual_output)
        
        # # Use a stubbed agent response (per question) to avoid waiting for live model calls.
        # stub = stub_map.get(question, {})
        # stub_answer = stub.get("answer", HotpotQA_expected_output)
        # stub_titles = stub.get("titles", supporting_titles)
        # stub_sentences = stub.get("sentences", HotpotQA_context)
        # stub_times = stub.get("times_used", item.get("wiki_times", 1))

        # actual_output = json.dumps({
        #     "answer": stub_answer,
        #     "tool_used": [
        #         {
        #             "tool": "Wikipedia",
        #             "times_used": stub_times,
        #             "titles": stub_titles,
        #         }
        #     ],
        #     "supporting_titles": stub_titles,
        #     "supporting_sentences": stub_sentences,
        #     "reasoning_explanation": [
        #         {
        #             "step": 1,
        #             "logic": f"Wikipedia shows evidence related to: {stub_answer}."
        #         }
        #     ],
        # })
        # agent_tool_usage_times = {"Wikipedia": stub_times}




        # Parse the agent JSON output to fill fields
        try:
            agent_response_json = json.loads(actual_output)
        except (json.JSONDecodeError, TypeError):
            agent_response_json = {}

        agent_final_answer = (
            agent_response_json.get("answer")
            or agent_response_json.get("final_answer")
            or actual_output
        )
        agent_supporting_sentences = agent_response_json.get("supporting_sentences", [])
        agent_supporting_titles = agent_response_json.get("supporting_titles", []) or agent_response_json.get(
            "wikipedia_titles_used", []
        )

        tool_used_field = agent_response_json.get("tool_used", [])
        agent_tools_used = []

        if isinstance(tool_used_field, str):
            agent_tools_used.append(ToolCall(name=tool_used_field, input_parameters={}))
        elif isinstance(tool_used_field, list):
            for entry in tool_used_field:
                tool_name = entry.get("tool") if isinstance(entry, dict) else None
                times_used = entry.get("times_used", 1) if isinstance(entry, dict) else 1
                titles = entry.get("titles", []) if isinstance(entry, dict) else []
                if tool_name:
                    agent_tools_used.append(
                        ToolCall(
                            name=tool_name,
                            input_parameters={"titles": titles} if titles else {},
                        )
                    )
                    # prefer explicit times_used if provided
                    if times_used:
                        agent_tool_usage_times[tool_name] = times_used
        # If parsing failed to yield tool calls, fall back to counted usage
        if not agent_tools_used and agent_tool_usage_times:
            for tool_name, times_used in agent_tool_usage_times.items():
                agent_tools_used.append(ToolCall(name=tool_name, input_parameters={}))
                
        
        test_case = LLMTestCase(
            input=question,
            actual_output=agent_final_answer,                
            expected_output=HotpotQA_expected_output,                
            retrieval_context=agent_supporting_sentences,  
            context= HotpotQA_context,
            tools_called= agent_tools_used,
            expected_tools= HotpotQA_tools_used,
            additional_metadata={
                "expected_facts": HotpotQA_context,
                "tool_usage":  agent_tool_usage_times,
                "expected_tool_usage": HotpotQA_expected_tools,
                "supporting_titles": supporting_titles, 
            }
            
        )

        # Debug logging for each constructed test case
        print("----- TEST CASE -----")
        print(f"Question: {question}")
        print(f"Expected answer: {HotpotQA_expected_output}")
        print(f"Actual answer: {agent_final_answer}")
        print(f"Expected tools: {HotpotQA_expected_tools}")
        print(f"Actual tools: {agent_tool_usage_times}")
        print(f"Expected facts: {HotpotQA_context}")
        print(f"Actual facts: {agent_supporting_sentences}")
        print(f"Expected tools detail: {HotpotQA_tools_used}")
        print(f"Actual tools detail: {agent_tools_used}")
        print("---------------------")

        test_cases.append(test_case)

    return test_cases



@pytest.mark.asyncio
async def test_rag() -> None:
    # Run evaluation and get test cases
    global test_cases_num
    test_cases = await create_rag_test_cases(test_cases_num) #number beqtween 1 and 50
    # Use local Ollama model for evaluation by default (no env key required)
    eval_model_name = os.environ.get("EVAL_CHAT_MODEL_NAME", "ollama:llama3.1:8b")

    # Increase DeepEval per-task timeout for local models (in seconds)
    os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "300")


    eval_model = DeepEvalLLM.from_name(eval_model_name)
    # RAG-specific metricszv
    contextual_relevancy = FaithfulnessMetric(
        model = eval_model,
        threshold=0.7
    )
    contextual_precision = AnswerRelevancyMetric(
        model = eval_model,
        threshold=0.7
    )
    tool_correctness_metric = ToolCorrectnessMetric(
        model=eval_model,
        include_reason=False,
    )
    ######### final answer
    # Metric 1: Ensure the final answer exactly matches the expected answer
    answer_exact_match_metric = ExactMatchMetric(threshold=1.0)

    # Metric 2: Ensure the final answer with llm as a judge
    answer_llm_judge_metric = AnswerLLMJudgeMetric(
        model=eval_model,
        threshold=0.7,
    )

    ######### tools
    # Metric 3: Compare tool usage and count vs expected tool usage and count
    tool_usage_metric = ToolUsageMetric()

    # Metric 4: Compare tool arguments
    argument_metric = ArgumentCorrectnessMetric(
        threshold=0.7,
        model = eval_model,
        include_reason=True
    )

    ######### supporting facts

    # Metric 5: Compare retrieved supporting sentences with expected facts - llm as a judge
    facts_metric = FactsSimilarityMetric(
        model=eval_model
    )    

    # Metric 6: measures how much of the truly relevant context (expected_facts / ground-truth evidence) the retrieved context covers.
    contextual_recall_metric = ContextualRecallMetric(
        model = eval_model,
        threshold=0.7
    )
    
    

    # Collect metrics to run (enable all for full table output)
    # Ordered by category:
    # Final answer metrics first, then tool metrics, then facts/context.
    metrics = [
        # Final answer
        answer_exact_match_metric,
        answer_llm_judge_metric,
        contextual_precision,
        contextual_recall_metric,
        contextual_relevancy,
        # Tools
        tool_correctness_metric,
        tool_usage_metric,
        argument_metric,
        # Facts / context
        facts_metric,
    ]

    # Evaluate using DeepEval incrementally
    all_test_results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Evaluating test case {i+1}/{len(test_cases)}...")
        try:
            # Run evaluation for a single test case
            res = evaluate(
                test_cases=[test_case], 
                metrics=metrics,
                display_config=DisplayConfig(
                    show_indicator=False, 
                    print_results=False, 
                    verbose_mode=False
                )
            )
            
            # Extract results and add to our collection
            step_results = (
                getattr(res, "results", None)
                or getattr(res, "test_results", None)
                or []
            )
            all_test_results.extend(step_results)
            
            # Pickle the accumulated results after each test case
            try:
                raw_path = Path("eval_results_raw.pkl")
                with raw_path.open("wb") as f:
                    pickle.dump(all_test_results, f)
                logger.info(f"Test case {i+1} saved to {raw_path}")
            except Exception as pickle_exc:
                logger.error(f"Failed to pickle after test case {i+1}: {pickle_exc}")
                
        except Exception as eval_exc:
            logger.error(f"Error evaluating test case {i+1}: {eval_exc}")
            traceback.print_exc()

    # Build and print the evaluation results table
    table = create_evaluation_table(all_test_results, metrics)
    print_evaluation_table(table)
    
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag())


