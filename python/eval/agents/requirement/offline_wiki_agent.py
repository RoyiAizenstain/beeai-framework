import json
import os
import sys
from pathlib import Path
import asyncio
import traceback
import tempfile
from typing import Counter, List
from beeai_framework.tools.tool import Tool
import pytest
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
)

from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from AnswerLLMJudgeMetric import AnswerLLMJudgeMetric
from ToolUsageMetric import ToolUsageMetric
from FactsSimilarityMetric import FactsSimilarityMetric
load_dotenv()

# Add the examples directory to sys.path to import setup_vector_store
examples_path = Path(__file__).parent.parent.parent.parent / "examples" / "agents" / "experimental" / "requirement"
sys.path.insert(0, str(examples_path))

from examples.agents.experimental.requirement.rag import setup_vector_store

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel, ToolMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool
from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search.wikipedia import (WikipediaTool)
from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.tools.weather import OpenMeteoTool

from beeai_framework.tools.code import PythonTool, LocalPythonStorage

from eval.model import DeepEvalLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric, ArgumentCorrectnessMetric

import pandas as pd

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

    model_name = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    llm = GeminiChatModel(model_name=model_name, api_key=os.environ.get("GEMINI_API_KEY"), allow_parallel_tool_calls=True )

    # Create RequirementAgent with multiple tools
    # tools: WikipediaTool for general knowledge, PythonTool for calculations, OpenMeteoTool for weather data

    #Format in Jason:
    #Final answer 
    #List of supporting sentences
    #explanation of reasoning for each sentence by its number
    #tool that was used
    #
    JSON_SCHEMA_STRING = """{"final_answer": "...","tool_used": "...","supporting_sentences": ["<sentence 1>", "<sentence 2>"],"reasoning_explanation": [{"step": 1, "logic": "The reasoning step"}]}"""
    JSON_SCHEMA_STRING = """{"final_answer": "...","tool_used": [{"tool": "...", "times_used": }]},"supporting_sentences": ["<sentence 1>", "<sentence 2>"],"reasoning_explanation": [{"step": 1, "logic": "The reasoning step"}]}"""
    JSON_SCHEMA_STRING = """{"final_answer": "...","tool_used": "...","supporting_sentences": ["<sentence 1>", "<sentence 2>"],"reasoning_explanation": [{"step": 1, "logic": "The reasoning step"}]}"""
    
    agent = RequirementAgent(
        llm=llm, 
        tools=[wiki_tool,OpenMeteoTool(), calculator_tool],
        memory=UnconstrainedMemory(),
        role="You are an expert Multi-hop Question Answering (QA) agent. Your primary role is to extract and combine information from the provided context to answer the user's question. Answer in jason format only.",
        instructions=[
            "RULES and CONSTRAINTS:",
            "1. SOURCE ADHERENCE (NO HALLUCINATION): Your final answer MUST be based ONLY on the context you retrieve from the provided tools (VectorStoreSearchTool or WikipediaTool). Do not use external knowledge.",
            "2. MULTI-HOP: You must perform multi-step reasoning or use multiple tools/retrievals if the question requires it.",
            "3. FINAL FORMAT: Your ONLY final output MUST be a single, valid JSON object adhering strictly to the required keys. Do not include any text outside the JSON block.",
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

async def create_rag_test_cases():
    """
    Create RAG test cases by directly invoking the agent and extracting retrieval context.
    """
    agent = await create_agent()
    
    test_cases = []
    
    output_json = """{"question": "Which magazine was started first Arthur's Magazine or First for Women?","answer": "Arthur's Magazine","relevant_sentences": ["Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.","First for Women is a woman's magazine published by Bauer Media Group in the USA."]}"""
    
    test_data = [
        {
            "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
            "answer": "Delhi",
            "relevant_sentences": [
            "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.",
            "The Oberoi Group is a hotel company with its head office in Delhi."
            ],
            "wiki_times": 2,
            "supporting_titles": [
            "Oberoi family",
            "The Oberoi Group"
            ]
        }
    ]
    for item in test_data:
        question = item["question"]
        
        HotpotQA_expected_output = item["answer"]
        HotpotQA_context = item["relevant_sentences"]
        HotpotQA_expected_tools = {"Wikipedia": item["wiki_times"]}
        supporting_titles = item["supporting_titles"]
        HotpotQA_tools_used = []
        for name in supporting_titles:
            HotpotQA_tools_used.append(ToolCall(name="Wikipedia", input_parameters={'query': name}))
        
        
        ## Run the agent
        #response = await agent.run(question)
        #actual_output = response.result.text
        #agent_tool_usage_times = count_tool_usage(response.memory.messages)
        agent_response =  """{"final_answer": "The head office of The Oberoi Group is in New Delhi, India.", "tool_used": "Wikipedia", "supporting_sentences": ["The Oberoi Group is a luxury hotel group with its head office in New Delhi, India."], "reasoning_explanation": [{"step": 1, "logic": "The Wikipedia search revealed that The Oberoi Group is a luxury hotel group with its head office in New Delhi, India."}]}"""
        agent_response_json = json.loads(agent_response)
        
        agent_final_answer = agent_response_json.get("final_answer", "")
        agent_supporting_sentences = agent_response_json.get("supporting_sentences", [])
        agent_tools_used = [
            ToolCall(name="Wikipedia", input_parameters={'full_text': True, 'query': 'Oberoi Hotels & Resorts'})
           
        ]
        agent_tool_usage_times = {"Wikipedia": 2}
                
        
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
        test_cases.append(test_case)

    return test_cases



@pytest.mark.asyncio
async def test_rag() -> None:
    # Run evaluation and get test cases
    test_cases = await create_rag_test_cases()
     # Get the evaluation model name from the environment, with a safe default
    eval_model_name = os.environ.get("EVAL_CHAT_MODEL_NAME", "google:gemini-2.5-flash")

    eval_model = DeepEvalLLM.from_name(eval_model_name)
    # RAG-specific metrics
    contextual_relevancy = FaithfulnessMetric(
        model = eval_model,
        threshold=0.7
    )
    contextual_precision = AnswerRelevancyMetric(
        model = eval_model,
        threshold=0.7
    )
    tool_correctness_metric = ToolCorrectnessMetric(
        include_reason=False
    )
    ######### final answer
    # Metric 1: Ensure the final answer exactly matches the expected answer
    #answer_exact_match_metric = ExactMatchMetric(threshold=1.0)

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
    
    

    # Evaluate using DeepEval
    eval_results = evaluate(
        test_cases=test_cases,
        #metrics=[contextual_precision, contextual_recall_metric, contextual_relevancy, tool_correctness_metric,argument_metric, facts_metric, tool_usage_metric, answer_llm_judge_metric],
        metrics=[argument_metric]
    )
    
    
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag())


