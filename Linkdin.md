Testing whether an AI agent "works" is a different problem than unit testing a function. An agent's output is prose, its correctness is partly semantic, and the path it took to an answer matters as much as the final text.

As part of my internship at IBM Research, I contributed an evaluation module to BeeAI Framework, IBM's open-source multi-agent toolkit (3.5k+ stars on GitHub, 26+ LLM providers) — so anyone building agents can score them systematically instead of eyeballing transcripts.

Rather than build yet another metrics framework, we leaned on two that already solve this well: DeepEval and Ragas. We added native LLM-as-judge support so their judge metrics can call any model BeeAI already supports (OpenAI, Anthropic, Ollama, VertexAI, Bedrock...), no separate API keys or client setup.

The real insight: a single pass/fail check misses most of what matters.

Take a multi-hop question like "Which magazine was started first, Arthur's Magazine or First for Women?" We score three things:
→ Is the answer right?
→ Did the agent call the tools it actually needed?
→ Did it cite the right supporting facts?

Two agents can land on the same correct answer and deserve very different scores once you look at how they got there. A low score on "facts cited" with a correct answer is the one to take most seriously: right conclusion, ungrounded reasoning.

Ships as a full working example: bundled dataset, agent, ready-to-run scripts for both frameworks. All merged and public today.

Full writeup in the comments 👇

Work done under the guidance of Antonpuz (IBM Research Israel), with Omri Meroz, reviewed by Tomáš Dvořák.

#AI #LLM #OpenSource #AgentEvaluation

---

**First comment:**
Full writeup: https://medium.com/@royi.aizen/evaluating-beeai-agents-building-a-deepeval-ragas-integration-0b2e071fcc20
PR: https://github.com/i-am-bee/beeai-framework/pull/1453

**Attach to post:** the input→agent→output→evaluation diagram.

**Note:** double-check the GitHub star count right before posting — it moves.
