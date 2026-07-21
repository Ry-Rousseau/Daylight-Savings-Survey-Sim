"""Full-transparency trace of the Stage-2 pipeline for the first 3 personas.

Wraps the real clients so every LLM call's exact (system, user, output) is captured,
runs the actual build_persona path, and writes a readable markdown trace.
"""
import json

import pandas as pd

from polis import prompts
from polis.llm import LLMClient, LLMConfig
from polis.persona_pipeline import (
    REFLECTION_COLUMNS,
    _PROVENANCE_COLUMNS,
    _parse_reflection,
    build_persona,
    describe_demographics,
    location_of,
    reflection_lines,
    sanitize_row,
    statement_memories,
)

DEMO_COLS = ["person_id", "state_name", "region", "sex", "age", "race_ethnicity",
             "hispanic_origin", "education", "occupation", "industry", "marital_status",
             "presence_of_children", "family_size"]
DONOR_COLS = list(REFLECTION_COLUMNS) + ["party_id_7pt_anes", "ideology_7pt_anes",
                                         "opinion_strength_anes"]


class TracingClient:
    def __init__(self, inner):
        self.inner = inner
        self.config = inner.config
        self.log = []

    def complete(self, messages, *, temperature=None, max_tokens=None):
        out = self.inner.complete(messages=messages, temperature=temperature, max_tokens=max_tokens)
        self.log.append({"system": messages[0]["content"], "user": messages[1]["content"],
                         "temperature": temperature, "output": out})
        return out


rows = pd.read_csv("data/persona_seed_dataset_n100.csv").to_dict("records")[:3]

bs = TracingClient(LLMClient(LLMConfig(model="qwen/qwen3-235b-a22b", reasoning=False, max_tokens=768)))
rf = TracingClient(LLMClient(LLMConfig(model="qwen/qwen3.7-max", reasoning=False, max_tokens=1024)))

seeded = [build_persona(r, backstory_client=bs, reflection_client=rf, seed=42) for r in rows]


def fence(s):
    return "```\n" + str(s).rstrip() + "\n```"


def kv(row, cols):
    return "\n".join(f"- **{c}**: {row.get(c)}" for c in cols if c in row and pd.notna(row.get(c)))


out = ["# Stage-2 pipeline trace — 3 personas\n",
       "backstory model: `qwen/qwen3-235b-a22b` (temp 0.9) · "
       "reflection model: `qwen/qwen3.7-max` (temp 0.3)\n"]

for i, (row, sp) in enumerate(zip(rows, seeded)):
    out.append(f"\n---\n\n# Persona {i+1} — `{sp.persona.id}` ({sp.persona.location})\n")

    out.append("## Stage 0 — raw inputs (from the joint dataset)\n")
    out.append("**ACS demographics (directly sampled):**\n" + kv(row, DEMO_COLS) + "\n")
    out.append("**Donor-matched disposition items (real, model-free):**\n" + kv(row, DONOR_COLS) + "\n")
    prov = "\n".join(
        f"- **{name}**: respondent `{row.get(idc)}`, match_distance `{row.get(dc)}`"
        for name, idc, dc in _PROVENANCE_COLUMNS if pd.notna(row.get(idc))
    )
    out.append("**Donor provenance (R29):**\n" + prov + "\n")

    clean, resolved = sanitize_row(row)
    out.append("\n## Stage 0.5 — sanitize (bake out cross-donor artifacts)\n")
    out.append("**resolved**: " + (str(resolved) if resolved else "none") + "\n")

    out.append("\n## Stage 1 — model-free rendering (no LLM, on the sanitized row)\n")
    out.append(f"**describe_demographics** → `{describe_demographics(clean)}`\n")
    out.append(f"**location** → `{location_of(clean)}`\n")
    out.append("**reflection_lines** (measures fed to BOTH LLM steps):\n"
               + "\n".join(f"- {l}" for l in reflection_lines(clean)) + "\n")
    out.append("**statement_memories** (first-person, model-free → seeded to memory):\n"
               + "\n".join(f"- {s}" for s in statement_memories(clean)) + "\n")

    out.append("\n## Stage 2a — BACKSTORY  (input → output)\n")
    out.append("**SYSTEM:**\n" + fence(bs.log[i]["system"]))
    out.append("**USER (input):**\n" + fence(bs.log[i]["user"]))
    out.append("**OUTPUT (backstory):**\n" + fence(bs.log[i]["output"]))

    out.append("\n## Stage 2b — REFLECTION → anchor  (input → output)\n")
    out.append("**SYSTEM:**\n" + fence(rf.log[i]["system"]))
    out.append("**USER (input — note: the MEASURES, not the backstory):**\n" + fence(rf.log[i]["user"]))
    out.append("**RAW OUTPUT (JSON):**\n" + fence(rf.log[i]["output"]))
    out.append(f"**PARSED** → values / dispositions:\n{fence(_parse_reflection(rf.log[i]['output']))}")

    out.append("\n## Stage 3 — final SeededPersona (what the agent gets)\n")
    out.append(f"- **description**: {sp.persona.description}")
    out.append(f"- **location**: {sp.persona.location}")
    out.append(f"- **values (R7 anchor)**: {list(sp.persona.values)}")
    out.append(f"- **dispositions (R7 anchor)**: {list(sp.persona.dispositions)}")
    out.append("- **seed memories @ t=0** (importance / created_at):")
    for m in sp.memories:
        out.append(f"    - (imp={m.importance}, t={m.created_at}) {m.text}")
    out.append(f"- **data_flags (measure contradictions, R29 hygiene)**: {sp.provenance.get('data_flags', [])}")
    out.append("- **provenance.generation**:\n" + fence(json.dumps(sp.provenance["generation"], indent=2)))

path = "sandbox/pipeline_trace_3personas.md"  # re-run rewrites this in-repo
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote", path)
print("empty-anchor:", [sp.persona.id for sp in seeded if not sp.persona.values and not sp.persona.dispositions] or "none")
