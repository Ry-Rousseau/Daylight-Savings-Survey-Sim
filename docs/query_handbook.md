# Query handbook — persona LLM endpoint

How to call the persona model. **Phases 0–2 (now): OpenRouter** — a managed OpenAI-compatible endpoint. **Phases 5+: self-hosted vLLM** (`Qwen/Qwen3-8B-AWQ`) on a remote GPU host. Rationale: ADR 0001 + ADR 0002. Both are OpenAI-compatible, so `src/polis/llm.py` is unchanged across the switch.

## Current backend — OpenRouter (Phases 0–2)

- Base URL `https://openrouter.ai/api/v1` · model `qwen/qwen3-8b` · key `OPENROUTER_API_KEY` in `.env`.
- **Disable Qwen3 reasoning for survey answers:** `extra_body={"reasoning": {"enabled": False}}` — with it on, a JSON schema can derail into unclosed output, and it burns tokens (ADR 0003).
- **Structured single-select:** `response_format={"type":"json_schema","json_schema":{"name":...,"strict":True,"schema":{...enum...}}}`. On OpenRouter this is *soft*-enforced → validate `choice ∈ options` and retry (`LLMClient.choose`). The hard grammar (`guided_choice`/`guided_json`) comes with self-hosted vLLM below.

```python
from polis import LLMClient
res = LLMClient().choose(system="You are a New Yorker.",
                         user="DST view? Choose one: ...", options=[...])
# -> {"choice", "reason", "model", "usage"}
```

---

## Self-hosted vLLM (Phases 5+)

The sections below describe the P5+ path — running `Qwen/Qwen3-8B-AWQ` under vLLM on a remote GPU host for the reportable runs.

## Endpoint
- Base URL: `http://<host>:8000/v1` — `<host>` is whatever public URL/IP the provider gives you. **Ephemeral — set per session**, don't hardcode.
- Auth: none by default → `api_key="not-needed"` (unless the server was started with `--api-key`).
- Configure via env: `POLIS_LLM_BASE_URL`, `POLIS_LLM_MODEL=Qwen/Qwen3-8B-AWQ`. Pin & log the model + server flags per run (R6).

## Serve (once, on the remote GPU host)
```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --host 0.0.0.0 --port 8000
```

## Query (from the agent / local machine)
```bash
curl http://<host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B-AWQ","messages":[{"role":"user","content":"..."}],"max_tokens":512}'
```
```python
from openai import OpenAI

client = OpenAI(base_url="http://<host>:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B-AWQ",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

## Thinking mode
Qwen3 ships a reasoning ("thinking") mode that may be on by default. For plain non-reasoning output, pass:
```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

## Parameters for future engineering

### Standard sampling (OpenAI-compatible)
- `temperature`, `top_p` — **per-agent fields, never a global constant (R1).**
- `max_tokens` — response cap.
- `seed` — reproducibility; log with the run (R17).
- `stop`, `presence_penalty`, `frequency_penalty`, `n`, `logprobs`.

### vLLM extras (via `extra_body`)
- `top_k`, `min_p`, `repetition_penalty`.
- `chat_template_kwargs` — e.g. `{"enable_thinking": False}`.

### Structured / constrained output (the action-vocabulary layer)
vLLM guarantees schema validity *by construction* (guided decoding), removing the parse-and-retry loop. Pass via `extra_body`:
- `guided_choice: [...]` — output is exactly one of the listed strings. **Ideal for single-select survey answers.**
- `guided_json: <JSON schema>` — e.g. `MyModel.model_json_schema()`; for structured actions like `{"action": "message", "target": "...", "content": "..."}`.
- `guided_regex`, `guided_grammar` — regex / EBNF grammar constraints.
- `guided_decoding_backend` — `xgrammar` (default) / `outlines` / `lm-format-enforcer`.

Example — single-select survey answer:
```python
resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B-AWQ",
    messages=[{"role": "user", "content": survey_prompt}],
    temperature=persona.temperature,          # per-agent (R1)
    extra_body={
        "guided_choice": question.options,     # answer ∈ question.options, guaranteed
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
answer = resp.choices[0].message.content
```

## Notes
- Host is ephemeral — expect cold starts / teardown; `base_url` is config, not a constant.
- Pin & log `model` + server flags per run (R6); log `seed`/params for reproducibility (R17).
- The local RTX 4070Ti (16 GB) is reserved for lightweight local models (e.g. the Phase 1 embedding model), not the persona LLM.
