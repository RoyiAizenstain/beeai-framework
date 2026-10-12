# Evaluating BeeAI Agents: Building a DeepEval + Ragas Integration

Testing whether an AI agent "works" is a different problem than testing whether a function returns the right value. An agent's output is prose, its correctness is partly semantic, and its path to an answer (which tools it called, in what order, with what arguments) matters as much as the final text.

[BeeAI Framework](https://github.com/i-am-bee/beeai-framework) is an open-source toolkit for building multi-agent systems in Python and TypeScript: agents that reason, call tools, and collaborate, backed by a unified interface across 26+ LLM providers (OpenAI, Anthropic, Ollama, VertexAI, Bedrock, and more). We recently added an evaluation module to it (currently Python-only) that lets you score agents systematically instead of eyeballing transcripts. I want to walk through the actual design decisions behind it, not just what it does but why it's shaped the way it is.

## Why not build our own evaluation framework?

The first temptation with any framework is to invent your own metrics system. We didn't, for a simple reason: [DeepEval](https://docs.confident-ai.com/) and [Ragas](https://docs.ragas.io/) already solve the hard parts, and solve them well.

DeepEval is a general-purpose LLM evaluation library, think "pytest for LLM outputs," with metrics for answer relevancy, faithfulness, tool correctness, and an easy path to writing your own. Ragas focuses specifically on retrieval-augmented and agentic pipelines, with metrics built around the idea that an answer is only as good as the context it was grounded in. Both are async-native, both support LLM-as-judge scoring (having a model grade another model's output against a rubric), and both already carry years of tuning on things like "how do you phrase a judge prompt so the score is stable across runs." Reinventing that ourselves would have meant reinventing all of it, badly, from scratch.

So instead, we've added native LLM-as-judge (LLMaaJ) support to BeeAI: a way for these frameworks' judge metrics to call *any* model BeeAI already supports (Ollama, OpenAI, VertexAI, Bedrock, whatever a user has configured), instead of requiring a separate, framework-specific API key and client setup. That's a bridge, not a framework. So that's what we built: two adapters.

## Two adapters, for maximal robustness

It would be tidy if `DeepEvalLLM` and `InstructorRagasLLM` shared a common base class. They don't, and that's deliberate: the two frameworks define genuinely different LLM contracts, and papering over that difference would produce a leakier abstraction than just writing two small, honest adapters.

DeepEval's `DeepEvalBaseLLM` expects `generate()`/`a_generate()` to return a **string**:

```python
class DeepEvalLLM(DeepEvalBaseLLM):
    async def a_generate(self, prompt: str, schema: TSchema | None = None) -> str:
        response = await self._model.run(
            [UserMessage(prompt)],
            response_format=schema.model_json_schema(mode="serialization") if schema else None,
        )
        return response.get_text_content()
```

Ragas's `InstructorBaseRagasLLM`, on the other hand, expects a **validated Pydantic object** back. It's built around the [Instructor](https://github.com/instructor-ai/instructor) pattern of structured extraction:

```python
class InstructorRagasLLM(InstructorBaseRagasLLM):
    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        response = await self.model.run([UserMessage(prompt)], response_format=response_model)
        if not isinstance(response.output_structured, response_model):
            raise ChatModelError(
                "The model failed to produce structured output matching the requested schema.",
                context={"output": response.get_text_content()},
            )
        return response.output_structured
```

Both adapters lean on the same underlying mechanism: BeeAI's `ChatModel.run(..., response_format=...)`, which already validates and repairs structured output internally. But they hand back fundamentally different shapes because that's what each framework actually needs. An early version of the Ragas adapter manually extracted JSON from the raw text with string slicing and a bit of regex. Once we noticed `response.output_structured` already did exactly this, more robustly, the manual extraction just disappeared: fewer lines, and one less thing that could silently break on a slightly-off model response.

![One ChatModel, two adapters: DeepEval accepts text or an optional schema, Ragas always expects a validated Pydantic object.](./images/two-adapters-diagram.png)

## What "evaluating an agent" actually means

Evaluating an agent means running it against a set of known test cases, questions where you already know a correct (or acceptable) answer, and then scoring what comes back. In its simplest form, that's just a dataset: a list of `(question, expected_answer)` pairs and a way to check the agent's actual answer against the expected one.

But an agent isn't a pure function. It reasons over multiple steps, decides which tools to call and in what order, and produces prose rather than a single deterministic value. So "did it get the right answer" is only one axis worth scoring. Did it call the tools it needed to, and skip the ones it didn't? Did it cite the right supporting facts, or land on the right answer for the wrong reason? Two agents can produce the identical final text and still deserve very different scores once you look at how they got there.

That's the actual shape of the problem BeeAI's evaluation module is built to handle: define a dataset once, run an agent against it, and score the result from several angles at once (answer correctness, tool usage, factual grounding) rather than a single pass/fail check.

## A complete, runnable example

Adapters and metrics are only useful if there's something real to point them at, so alongside the adapters we built a full working example: an agent, a bundled dataset, and two ready-to-run experiment scripts (one for DeepEval, one for Ragas) that evaluate the same agent against the same questions. It's meant to be the thing you actually run first, before you build your own dataset or agent:

```bash
pip install "beeai-framework[evaluation]"
cd python/examples/evaluation
EVAL_CHAT_MODEL_NAME=openai:gpt-4o python deepeval/experiment.py
```

That runs the full bundled dataset through the agent and prints a pass/fail summary with a score per metric. Swap `deepeval/experiment.py` for `ragas/experiment.py` (after a one-time `python ragas/dataset.py` to build its dataset file) to see the same questions scored through Ragas instead.

### Why HotpotQA for the bundled dataset

The example dataset needed to exercise the interesting part of agent evaluation: multi-step reasoning across separate information sources, not single-lookup Q&A. [HotpotQA](https://hotpotqa.github.io/) is a well-known multi-hop QA benchmark built exactly for this. Each question requires combining facts from two or more separate Wikipedia articles.

Rather than write synthetic questions, the bundled example uses two real HotpotQA records, verified against the actual dataset, not just written in a similar style. But HotpotQA's own schema carries more than an agent evaluation needs: a `context` field with every sentence from every candidate article, including irrelevant "distractor" paragraphs meant to test retrieval precision in the original benchmark's own use case (which isn't ours), plus a `supporting_facts` field that points into `context` by `title` + `sent_id` index rather than carrying the actual sentence text.

We simplified this down to five fields an agent and its metrics can use directly: `question`, `expected_answer`, `supporting_sentences` (the pointed-to sentences resolved into real text, not indices), `supporting_titles`, and `expected_tool_calls` (how many distinct Wikipedia lookups the question requires, which has no HotpotQA equivalent since it's specific to scoring *our* agent's tool usage). The distractor paragraphs and HotpotQA's own difficulty/type metadata got dropped; they're not wrong to have, just not needed for this example's scope.

### Walking through the example

It helps to see this against one real question from the bundled dataset:

> Which magazine was started first, Arthur's Magazine or First for Women?

Neither answer is in the question, and no single Wikipedia article settles it either: you need the founding date of *Arthur's Magazine* and the founding date of *First for Women*, then a comparison. That's the multi-hop shape the whole example is built around: two independent lookups, combined into one answer.

The agent under evaluation is a `RequirementAgent` with three tools and a tightly-scoped role: `WikipediaTool` and `OpenMeteoTool` from the framework, plus a small example-only `CalculatorTool` defined alongside the example code itself.

```python
def create_agent(model_name: str | None = None) -> RequirementAgent:
    llm = ChatModel.from_name(model_name, {"allow_parallel_tool_calls": True})
    return RequirementAgent(
        llm=llm,
        tools=[WikipediaTool(), OpenMeteoTool(), CalculatorTool()],
        memory=UnconstrainedMemory(),
        role=(
            "You are an expert Multi-hop Question Answering (QA) agent. "
            "Your primary role is to extract and combine information from the provided context "
            "to answer the user's question. Answer in json format only."
        ),
        instructions=[
            "1. SOURCE ADHERENCE (NO HALLUCINATION): base your answer ONLY on tool output, "
            "never external knowledge.",
            "2. MULTI-HOP: perform multi-step reasoning or use multiple tools if the question requires it.",
            "3. FINAL FORMAT: respond with a single JSON object only: answer, tool_used, "
            "supporting_titles, supporting_sentences, reasoning_explanation.",
        ],
    )
```

`WikipediaTool` is the one doing real work here; `OpenMeteoTool` and `CalculatorTool` exist so the agent has a couple of other tools it *shouldn't* reach for on this kind of question. That matters, because one of the metrics scores whether it called the *right* tools, not just any tools.

The dataset entry for this question carries everything needed to grade the answer once it comes back:

```json
{
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "expected_answer": "Arthur's Magazine",
  "supporting_sentences": [
    "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia...",
    "First for Women is a woman's magazine published by Bauer Media Group in the USA."
  ],
  "expected_tool_calls": 2,
  "supporting_titles": ["Arthur's Magazine", "First for Women"]
}
```

Run the agent on this question, and a passing trace looks up both magazines on Wikipedia, compares their founding information, and returns `"Arthur's Magazine"` as the concise answer, with `supporting_titles` matching the two article titles it actually consulted. From there, the metrics take over: `AnswerLLMJudgeMetric` checks the answer is right (semantically, not just character-for-character), `ToolUsageMetric` checks it called Wikipedia twice as expected rather than guessing after one lookup, and `FactsSimilarityMetric` checks the cited sentences actually match what the dataset says justifies the answer. Three different questions about the same run, three different scores: that's the whole reason to run more than one metric in the first place.

![One run, three checks: each metric reads a different part of the trace and compares it to the HotpotQA ground truth.](./images/three-metrics-diagram.png)

## Where the built-in metrics stop, and why we wrote three more

DeepEval and Ragas ship strong general-purpose metrics: answer relevancy, faithfulness, context recall. What they don't ship is anything specific to a given agent's contract. Alongside the adapters, the bundled example under `examples/evaluation/` includes three example metrics that show the pattern, each filling a specific gap:

- **`AnswerLLMJudgeMetric`**: semantic equivalence between the agent's answer and the expected one, judged by an LLM rather than exact string matching (paraphrasing shouldn't count as wrong).
- **`ToolUsageMetric`**: whether the agent called the expected tools the expected *number* of times, which matters specifically for multi-hop questions where under-calling a tool (stopping after one Wikipedia lookup when two were needed) is a distinct failure mode from calling the wrong tool.
- **`FactsSimilarityMetric`**: semantic similarity between the facts the agent actually cited and the facts it was expected to cite, independent of whether the final answer text happened to match.

Each extends the relevant framework's own metric base class and uses the adapter to do its LLM-judge scoring, so a custom metric is just as portable across providers as a built-in one. Writing your own metric doesn't mean giving up the "use any BeeAI-supported model" property; it's the same adapter, reused.

## What a bad score tells you

The real value of running several metrics on the same trace is that each failure points somewhere different:

- **Low `AnswerLLMJudgeMetric`**: the most direct signal. The agent just got it wrong, time to look at the model, the prompt, or the question itself.
- **Low `ToolUsageMetric`, correct answer**: the agent got lucky. It skipped a lookup it needed or called a tool it didn't, worth fixing even though the final text looked fine.
- **Low `FactsSimilarityMetric`, correct answer**: the one to take most seriously. The agent landed on the right text while citing weak or unrelated supporting facts, a quieter form of the same problem as hallucination: right conclusion, ungrounded reasoning.

None of these show up if you only check whether the final answer string matches; that's the whole case for scoring from more than one angle.

## What this actually buys you

None of this is exotic engineering: it's two adapters, a simplified dataset, and three metrics. But that's kind of the point. The interesting problem in agent evaluation isn't inventing new statistics, it's making the *plumbing* boring enough that you can focus on what the numbers are actually telling you. `pip install "beeai-framework[evaluation]"`, wrap a `ChatModel`, and DeepEval or Ragas can judge your agent's answers using the same model config you already have. No separate API keys, no framework-specific client setup, no format mismatch to debug at 2am.

The adapters live at `beeai_framework.evaluation.adapters`, ready to import and reuse in your own metrics. The three example metrics above, along with the bundled multi-hop QA dataset and agent, live under `examples/evaluation/` in the repo — not in the installed package — if you want to see the whole thing wired together or use them as a starting point for your own.

This work was done under the guidance of Antonpuz (IBM Research Israel) and in collaboration with Omri Meroz 💪🏻

Thanks 👨‍💻 Tomáš Dvořák (Tomas2D) for the code review :)

GitHub: https://github.com/i-am-bee/beeai-framework/pull/1453
