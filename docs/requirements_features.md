The full architecture is on a very fundamental-level prompt-in-a-loop directed at a specific LLM endpoint. But everything around it requires detailed system design and local engineering. The current literature does converge on several genuine engineering subsystems that sit underneath the prompting, and they're the difference between "chained prompts" and something that scales, stays coherent, and produces auditable behavior.

# The non-LLM prompt features (liking with layers described in `design_layers.md`)

## 1. Embeddings-based memory retrieval (not just "context")

This is the most standardized non-prompting component in the field. The Park et al. generative agents architecture treats memory as a separate data structure, not a growing block of context text: every experience is stored as a discrete record, embedded via a text embedding model, and retrieval at decision-time combines three scored signals — **recency** (exponential time-decay), **importance** (an LLM-generated 1–10 salience score assigned once at write-time, not at retrieval-time), and **relevance** (cosine similarity between the embedding of the current situation and each stored memory). The scores are normalized and combined, usually with roughly equal weighting, and only the top-N records enter the prompt for that decision.

The technical reason this matters for you: without it, either your context grows unbounded (cost and coherence collapse) or you truncate arbitrarily (agents forget things that matter and remember things that don't). This is a real subsystem — vector store, embedding model, decay function, scoring pipeline — that runs *outside* the LLM call, not inside a prompt.

A layer on top of this worth knowing about: **reflection**. Periodically the agent is prompted with its most recent memory records to generate higher-level "insights," which get written back into the memory store as new, higher-importance memories that can themselves be retrieved later — producing a tree of increasingly abstract self-generated conclusions rather than a flat log. That's a meaningful architectural choice, not a prompt trick.

## 2. Structured/constrained output — turning free text into an executable action

If your agents emit free-form prose, your simulation engine has no reliable way to parse "what did the agent actually do" into a state transition — you're stuck regex-matching or hoping the model formats things consistently. The standard fix is **constrained decoding**: the token sampler itself is restricted at each generation step to only tokens that keep the output valid against a schema or grammar, so the model is structurally incapable of producing an invalid action. This is done via logit masking against a JSON schema or formal grammar (tools like Outlines, Guidance, or the XGrammar backend now built into vLLM), and it eliminates the retry-and-reparse loop entirely — the output is guaranteed schema-valid by construction, not by hope.

For your use case this is the mechanism that lets an agent's "decision" (expressed by the LLM in natural language reasoning) get compiled down into a small, fixed action vocabulary your engine can actually execute deterministically — e.g., `{"action": "message", "target": "agent_42", "content": "..."}` rather than parsing arbitrary prose.

## 3. A non-LLM world-state / action-resolution layer

"is it just prompts talking to prompts?" No, in the serious frameworks it explicitly isn't. Concordia's Game Master pattern is the clearest version: agents propose actions in natural language, but a separate component — which can be pure code, not an LLM — determines whether the action is legal, resolves its effects on world state, and updates the shared environment. The LLM decides *intent*; a symbolic layer decides *consequence*. This split is what gives you determinism, auditability, and prevents contradictory or physically-impossible cascades (two agents both "winning" the same auction, an agent referencing an event that never happened, etc.).

There's active work formalizing this further as hybrid symbolic-neural agent loops — separating retrieval, cognition, symbolic control, action, and memory into distinct phases specifically to restore the explainability and controllability that pure prompt-chaining lacks. The pattern name to search for if you want to go deeper here is "neurosymbolic" or "hybrid symbolic-LLM" agent architectures.



## Outcome features

Important: real humans don't always choose a side, a lot of human beings are ambivilent about a topic and need exposure / interaction for that to change. Weightiness and inertia in human opinion is real and should not be underestimated in design choices.

We also don't want convergence - this is primary design principles behind the principle laid out in `design_layers.md`
