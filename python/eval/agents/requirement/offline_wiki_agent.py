import json
import os
import sys
from pathlib import Path
import asyncio
import traceback
import tempfile
from typing import List
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

    agent = RequirementAgent(
        llm=llm, 
        tools=[wiki_tool,OpenMeteoTool(), calculator_tool],
        memory=UnconstrainedMemory(),
        role="You are an expert Multi-hop Question Answering (QA) agent. Your primary role is to extract and combine information from the provided context to answer the user's question. Answer concisely. Answer in jason format only.",
        instructions=[
            "RULES and CONSTRAINTS:",
            "1. SOURCE ADHERENCE (NO HALLUCINATION): Your final answer MUST be based ONLY on the context you retrieve from the provided tools (VectorStoreSearchTool or WikipediaTool). Do not use external knowledge.",
            "2. MULTI-HOP: You must perform multi-step reasoning or use multiple tools/retrievals if the question requires it.",
            "3. FINAL FORMAT: Your ONLY final output MUST be a single, valid JSON object adhering strictly to the required keys. Do not include any text outside the JSON block.",
            "4. THE JSON SCHEMA STRING: " + JSON_SCHEMA_STRING,
            "5. Answer the user's question directly and concisely."
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
    # Define test questions and expected outputs
    test_data = [
        (
            "Which magazine was started first Arthur's Magazine or First for Women?",
            output_json
        ),
        #(
         #    "Which magazine was started first Arthur's Magazine or First for Women?",
          #  bad_agent_json
        #)
        # (
        #     "What tools can be used with BeeAI agents?",
        #     "BeeAI agents can use various tools including Search tools (DuckDuckGoSearchTool), Weather tools (OpenMeteoTool), Knowledge tools (LangChainWikipediaTool), and many more available in the beeai_framework.tools module. Tools enhance the agent's capabilities by allowing interaction with external systems."
        # ),
        # (
        #     "What memory types are available for agents?",
        #     "Several memory types are available for different use cases: UnconstrainedMemory for unlimited storage, SlidingMemory for keeping only the most recent messages, TokenMemory for managing token limits, and SummarizeMemory for summarizing previous conversations."
        # ),
        # (
        #     "How can I customize agent behavior in BeeAI Framework?",
        #     "You can customize agent behavior in five ways: 1) Setting execution policy to control retries, timeouts, and iteration limits, 2) Overriding prompt templates including system prompts, 3) Adding tools to enhance capabilities, 4) Configuring memory for context management, and 5) Event observation to monitor execution and implement custom logging."
        # )
    ]
    
    for question, expected_output in test_data:
        # Run the agent
        #response = await agent.run(question)

        response = """{"final_answer": "Arthur's Magazine was started first. It was published from 1844 to 1846, while First for Women was started in 1989.", "tool_used": "Wikipedia", "supporting_sentences": ["Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.", "First for Women was an American woman's magazine published by McClatchey Media owned A360media. The magazine was started in 1989 by Bauer Media Group."], "reasoning_explanation": [{"step": 1, "logic": "I used the Wikipedia tool to find the start dates of both magazines."}, {"step": 2, "logic": "I compared the start dates and determined which magazine was started first."}] }"""
        #bad_response=  """{"final_answer": "Arthur's Magazine","tool_used": "Intuition","supporting_sentences": [],"reasoning_explanation": [{"step": 1, "logic": "I looked at the name 'Arthur' and it reminds me of King Arthur from the middle ages, so it sounds very old."},{"step": 2, "logic": "The name 'First for Women' sounds like a modern feminist movement, so it must be new."},{"step": 3, "logic": "Therefore, based on the vibes of the names, Arthur's Magazine is older."}]}"""
        #actual_output = response.result.text
        actual_output = response
        json_data = json.loads(actual_output)

        agent_final_answer = json_data.get("final_answer", "")
        agent_supporting_sentences = json_data.get("supporting_sentences", [])

        
        # Extract retrieval context from message history
        #retrieval_context = extract_retrieval_context(response.memory.messages)
        
        #reasoning_text = "\n".join([f"Step {step.step}: {step.logic}" for step in agent_output.reasoning_explanation])

        #actual_output = [response.result.text, json.loadstool_usage]
        actual_output =agent_final_answer

        HotpotQA_expected_output = json.loads(expected_output).get("answer", "")
        HotpotQA_context = json.loads(expected_output).get("relevant_sentences", [])
        # Create test case
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,                
            expected_output=HotpotQA_expected_output,                
            retrieval_context=agent_supporting_sentences,  
            context= HotpotQA_context  
        )                                 
        test_cases.append(test_case)

        #TODO trajectory check
    
    return test_cases

def create_trajectory_metric(model, threshold=0.7):
    trajectory_criteria = """
    Evaluate the 'Reasoning Trajectory' provided in the Actual Output.
    
    1. **Logical Flow:** Do the steps follow a logical sequence? (e.g., Step 1 leads to Step 2).
    2. **Tool Usage:** Did the agent decide to use a tool when it was necessary?
    3. **Grounding:** Is every reasoning step supported by the 'Retrieval Context'? The agent should not make assumptions outside of the provided text.
    4. **Relevance:** Do the reasoning steps directly address the user's 'Input'?
    5. **Conclusion:** Do the steps logically justify the 'Final Answer'?
    """
    return GEval(
        name="Trajectory Logic",
        criteria=trajectory_criteria,
        evaluation_params=[
            LLMTestCaseParams.INPUT, 
            LLMTestCaseParams.ACTUAL_OUTPUT, 
            LLMTestCaseParams.RETRIEVAL_CONTEXT
        ],
        model=model,
        threshold=threshold
    )

@pytest.mark.asyncio
async def test_rag() -> None:
    # Run evaluation and get test cases
    test_cases = await create_rag_test_cases()
     # Get the evaluation model name from the environment, with a safe default
    eval_model_name = os.environ.get("EVAL_CHAT_MODEL_NAME", "google:gemini-2.5-flash")

    eval_model = DeepEvalLLM.from_name(eval_model_name)
    # RAG-specific metrics
    contextual_recall = ContextualRecallMetric(
        model = eval_model,#DeepEvalLLM.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]),
        threshold=0.7
    )
    contextual_relevancy = FaithfulnessMetric(
        model = eval_model,#DeepEvalLLM.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]),
        threshold=0.7
    )
    contextual_precision = AnswerRelevancyMetric(
        model = eval_model,#DeepEvalLLM.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]),
        threshold=0.7
    )
    trajectory_metric = create_trajectory_metric(
        model = eval_model,
        threshold=0.7
    )

    #TODO: add more supporting sentences metric
    #TODO: add more reasoning steps metric
    #TODO: KEYs

    # Evaluate using DeepEval
    eval_results = evaluate(
        test_cases=test_cases,
        metrics=[contextual_precision, contextual_recall, contextual_relevancy, trajectory_metric],

    )
    print(eval_results)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag())


"""
Prompt:
I have two jason files. One is from HotpotQA dataset with context and answers.
The other is from an agent that uses retrieval augmented generation (RAG) to answer questions.
I want to compare the two files to see how well the agent performed compared to the ground truth answers in the HotpotQA dataset.
i want to use deepeval framework to compare the two files.
hotputqa file format:

[
  {
    "question": "Which magazine was started first Arthur's Magazine or First for Women?",
    "answer": "Arthur's Magazine",
    "relevant_sentences": [
      "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",
      "First for Women is a woman's magazine published by Bauer Media Group in the USA."
    ],
    "wiki_times": 2
  },
  {
    "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
    "answer": "Delhi",
    "relevant_sentences": [
      "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.",
      "The Oberoi Group is a hotel company with its head office in Delhi."
    ],
    "wiki_times": 2
  }


my agent:

{"final_answer": "Arthur's Magazine was started first. It was published from 1844 to 1846, while First for Women was started in 1989.", "tool_used": "Wikipedia", "supporting_sentences": ["Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.", "First for Women was an American woman's magazine published by McClatchey Media owned A360media. The magazine was started in 1989 
by Bauer Media Group."], "reasoning_explanation": [{"step": 1, "logic": "I used the Wikipedia tool to find the start dates of both magazines."}, {"step": 2, "logic": "I compared the start dates and determined which magazine was started first."}] }

which metrics from deepeval should i use to compare the two files and how?
"""