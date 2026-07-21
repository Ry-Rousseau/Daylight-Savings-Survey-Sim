"""Persona-seeding pipeline — Stage 2 (Layer 2 — Persona; ADR 0016, P6a).

Turns one row of the joint census+disposition dataset (``data/persona_seed_dataset_n100.csv``:
ACS demographics + donor-matched ANES/GSS/ATUS disposition items, each carrying its
real respondent id + match distance for R29 provenance) into a :class:`SeededPersona`
— a :class:`~polis.persona.Persona` plus the t=0 seed memories the agent starts with.

The bias discipline (ADR 0016): the disposition items are *real*, borrowed from a
matched survey respondent (model-free) — the LLM only **elaborates** them, never
invents them. Two seed-time LLM steps:

1. **backstory** — a short first-person narrative that *enacts* the fixed measures
   (a model can act a story more consistently than a table of scores).
2. **reflection → anchor** (Variant A) — synthesizes the always-on ``values`` /
   ``dispositions`` R7 anchor **from the real measures, not the backstory**, so the
   anchor stays grounded in real data rather than compounding model confabulation.

Placement is hybrid: the reflection populates the always-on persona anchor; the
backstory + demographic facts + first-person disposition statements are seeded into
the agent's private memory at t=0 (importance set at creation, recency at sim start).

The corpus is generated **once** (this is a build step, re-runnable from its notebook),
cached to JSON with a content hash + generation provenance (R17), and loaded at run
time by ``Population.from_corpus`` — so simulations pay no seed-time LLM cost and stay
reproducible against a pinned corpus.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .memory_seeds import SeedSpec
from .persona import Persona

logger = logging.getLogger(__name__)

CORPUS_VERSION = 1
BACKSTORY_PROMPT_V = "backstory_user/v1"
REFLECTION_PROMPT_V = "reflection_user/v1"

# --- disposition-item selection (ADR 0016 judgment call) -----------------------
#
# Which donor columns become persona content. Deliberately EXCLUDED by default:
# raw partisan identity (``party_id_7pt_anes``, ``ideology_7pt_anes``) and the custom
# ``opinion_strength_anes`` scale. DST is non-partisan; injecting explicit party/
# ideology as persona content would risk *re-stereotyping* the very voice this layer
# exists to de-stereotype, and the opinion-strength scale's range/direction is not
# reliably known here. All three remain in the dataset + provenance for an opt-in
# ablation — they are simply not spoken into the persona by default.

# Columns rendered as first-person *seed memories* (model-free, lived phrasing).
STATEMENT_COLUMNS: tuple[str, ...] = (
    "wake_time_atus",
    "usual_hours_worked_gss",
    "general_happiness_gss",
    "social_trust_gss",
    "tipi_extraversion_anes",
    "tipi_agreeableness_anes",
    "tipi_conscientiousness_anes",
    "tipi_emotional_stability_anes",
    "tipi_openness_anes",
    "religion_importance_anes",
)

# Columns summarized as *measures* fed to the backstory + reflection LLM steps.
REFLECTION_COLUMNS: tuple[str, ...] = STATEMENT_COLUMNS + (
    "moral_traditionalism_anes",
    "egalitarianism_anes",
    "religion_group_anes",
    "religion_attendance_anes",
)

# Donor provenance columns (R29): matched respondent id + match distance per source.
_PROVENANCE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("anes", "anes_respondent_id", "anes_match_distance"),
    ("gss", "gss_respondent_id", "gss_match_distance"),
    ("atus", "atus_respondent_id", "atus_match_distance"),
)


# --- value helpers -------------------------------------------------------------

def _missing(v: Any) -> bool:
    """True for an absent/NaN/ACS-``N/A`` value, without a hard pandas dependency."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and (not v.strip() or v.strip().startswith("N/A")):
        return True
    return False


def _get(row: Mapping[str, Any], key: str) -> Any:
    v = row.get(key)
    return None if _missing(v) else v


def _level(score: float, lo: float, hi: float) -> str:
    """Coarse low/moderate/high label for a numeric measure on a ``[lo, hi]`` scale."""
    span = hi - lo
    if score <= lo + span / 3:
        return "low"
    if score >= hi - span / 3:
        return "high"
    return "moderate"


def _clean_occupation(occ: str) -> str:
    """ACS occupation codes look like ``SAL-Sales Representatives Of Services…``; drop
    the leading code, lower-case, and trim to a short readable phrase."""
    text = occ.split("-", 1)[1] if "-" in occ else occ
    words = text.strip().split()
    return " ".join(words[:6]).lower().rstrip(" ,;:-")


# --- demographics -> factual description ---------------------------------------

def describe_demographics(row: Mapping[str, Any]) -> str:
    """A factual noun-phrase description composed from the ACS demographic fields —
    reads naturally after ``You are …`` and as ``I'm …``. Missing fields are skipped."""
    head: list[str] = ["a"]
    age = _get(row, "age")
    if age is not None:
        head.append(f"{int(age)}-year-old")

    marital = _get(row, "marital_status")
    if marital is not None:
        head.append(str(marital).lower())

    race = _get(row, "race_ethnicity")
    if race is not None:
        race = str(race).replace(" alone", "")
        hispanic = _get(row, "hispanic_origin")
        if hispanic is not None and not str(hispanic).startswith("Not Spanish"):
            race = f"Hispanic/Latino {race}"
        head.append(race)

    sex = _get(row, "sex")
    if sex is not None:
        head.append({"Female": "woman", "Male": "man"}.get(str(sex), str(sex).lower()))
    else:
        head.append("person")

    phrase = " ".join(head)

    clauses: list[str] = []
    edu = _get(row, "education")
    if edu is not None:
        clauses.append(f"with a {str(edu).lower()}")
    occ = _get(row, "occupation")
    if occ is None:
        clauses.append("who is not currently working")
    else:
        clauses.append(f"who works as {_clean_occupation(str(occ))}")

    return phrase + (" " + " ".join(clauses) if clauses else "")


def location_of(row: Mapping[str, Any]) -> str:
    """The persona's US locale (ADR 0015). State name preferred, then region, then a
    country fallback — never NYC, so a census population isn't silently all-NYC."""
    return str(_get(row, "state_name") or _get(row, "region") or "the United States")


# --- disposition items -> first-person memory statements (model-free) ----------

def _wake_statement(v: Any) -> str | None:
    m = re.match(r"^(\d{1,2}):(\d{2})", str(v))
    if not m:
        return None
    h = int(m.group(1))
    ampm = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    return f"On a typical day I'm up around {h12}{ampm}."


_TIPI_PHRASES = {
    "tipi_extraversion_anes": ("I'm outgoing and talkative", "I'm fairly reserved and quiet"),
    "tipi_agreeableness_anes": ("I try to be warm and easy to get along with", "I'm blunt and don't mind a bit of friction"),
    "tipi_conscientiousness_anes": ("I'm organized and like to plan ahead", "I'm easygoing about plans and details"),
    "tipi_emotional_stability_anes": ("I stay calm and don't rattle easily", "I get stressed and worry more than most"),
    "tipi_openness_anes": ("I'm curious and drawn to new ideas", "I prefer the familiar and what's worked before"),
}


def _tipi_statement(col: str, v: Any) -> str | None:
    level = _level(float(v), 1, 7)
    if level == "moderate":
        return None
    high, low = _TIPI_PHRASES[col]
    return (high if level == "high" else low) + "."


_TRUST_PHRASES = {
    "Can trust": "I generally think most people can be trusted.",
    "Cannot trust": "I think you can't be too careful dealing with people.",
    "Depends": "Whether people can be trusted, I figure it depends.",
}


def statement_memories(row: Mapping[str, Any]) -> list[str]:
    """First-person, model-free lived statements from the selected disposition items."""
    out: list[str] = []
    for col in STATEMENT_COLUMNS:
        v = _get(row, col)
        if v is None:
            continue
        if col == "wake_time_atus":
            s = _wake_statement(v)
        elif col == "usual_hours_worked_gss":
            # Cross-donor guard (ADR 0016 multi-donor caveat): only voice work hours if
            # the ACS record is actually employed — a GSS work-hours donor can otherwise
            # contradict a not-working ACS status (e.g. a retiree "working 40 hours").
            s = (
                f"I usually work about {int(v)} hours a week."
                if int(v) > 0 and _get(row, "occupation") is not None
                else None
            )
        elif col == "general_happiness_gss":
            s = f"These days I'd say I'm {str(v).lower()}."
        elif col == "social_trust_gss":
            s = _TRUST_PHRASES.get(str(v))
        elif col.startswith("tipi_"):
            s = _tipi_statement(col, v)
        elif col == "religion_importance_anes":
            # Consistency guard: the ANES religion items can conflict internally
            # (importance=Important while group=Not religious / attendance=Never). Only
            # voice faith as *lived* when the person is actually affiliated and attends,
            # so the memory doesn't contradict the group/attendance. The reflection step
            # still receives all three raw measures and reconciles them itself.
            important = str(v).lower().startswith("important") or "very" in str(v).lower()
            group = _get(row, "religion_group_anes")
            attendance = _get(row, "religion_attendance_anes")
            affiliated = group is not None and "not religious" not in str(group).lower()
            attends = attendance is None or str(attendance).lower() != "never"
            s = "My faith is an important part of my life." if important and affiliated and attends else None
        else:  # pragma: no cover - selection is closed above
            s = None
        if s:
            out.append(s)
    return out


# --- disposition items -> measure lines for the LLM steps ----------------------

def _measure_line(col: str, v: Any) -> str:
    if col.startswith("tipi_"):
        trait = col[len("tipi_"):].replace("_anes", "").replace("_", " ")
        return f"{trait.capitalize()} — {_level(float(v), 1, 7)} ({float(v):.1f} on 1-7)"
    if col == "moral_traditionalism_anes":
        return f"Moral traditionalism — {_level(float(v), 1, 5)} (1=progressive, 5=traditional)"
    if col == "egalitarianism_anes":
        return f"Egalitarianism — {_level(float(v), 1, 5)} (1=low, 5=high)"
    if col == "usual_hours_worked_gss":
        return f"Usual work hours per week — {int(v)}"
    if col == "wake_time_atus":
        return f"Typical wake time — {str(v)[:5]}"
    labels = {
        "general_happiness_gss": "General happiness",
        "social_trust_gss": "Social trust",
        "religion_group_anes": "Religion",
        "religion_attendance_anes": "Religious attendance",
        "religion_importance_anes": "Religion importance",
    }
    return f"{labels.get(col, col)} — {v}"


def reflection_lines(row: Mapping[str, Any]) -> list[str]:
    """Compact ``Name — level`` measure lines fed to the backstory + reflection LLM
    steps (the fixed inputs they must stay consistent with)."""
    return [_measure_line(c, _get(row, c)) for c in REFLECTION_COLUMNS if _get(row, c) is not None]


# --- measure-level data hygiene ------------------------------------------------

def measure_contradictions(row: Mapping[str, Any]) -> list[str]:
    """Flag internal contradictions *between measures* before they get baked into a
    persona (a validation check at construction time, not a silent reconciliation by the
    reflection step). These arise from fusing donors + source-instrument artifacts —
    e.g. a not-working ACS record carrying GSS work hours, or ANES religion importance
    that disagrees with affiliation/attendance. Surfaced in provenance (``data_flags``)
    and logged, so a persona that looks internally incoherent is explained by the data,
    not mistaken for a model failure."""
    flags: list[str] = []
    occ, hours = _get(row, "occupation"), _get(row, "usual_hours_worked_gss")
    if occ is None and hours is not None and int(hours) > 0:
        flags.append(f"ACS not-working but usual_hours_worked_gss={int(hours)}")
    imp = _get(row, "religion_importance_anes")
    group, att = _get(row, "religion_group_anes"), _get(row, "religion_attendance_anes")
    important = imp is not None and (str(imp).lower().startswith("important") or "very" in str(imp).lower())
    if important and (
        (group is not None and "not religious" in str(group).lower())
        or (att is not None and str(att).lower() == "never")
    ):
        flags.append(f"religion importance={imp} but group={group}/attendance={att}")
    return flags


def sanitize_row(row: Mapping[str, Any]) -> tuple[dict, list[str]]:
    """Bake out **cross-donor fusion artifacts** before generation — contradictions
    between measures borrowed from *different* donor surveys, where one source is
    authoritative. Returns a cleaned row copy + the resolutions applied.

    Currently: a not-working ACS record (directly sampled, authoritative) carrying GSS
    work hours (donor-borrowed) — no real person is both, so the donated hours are
    nulled. **Within-donor tensions are NOT touched** (e.g. the ANES religion items all
    come from one respondent — "important but unaffiliated/non-attending" is a real
    "spiritual but not religious" profile, not a fusion bug); those are flagged by
    :func:`measure_contradictions` and left for the reflection to reconcile."""
    clean = dict(row)
    resolved: list[str] = []
    occ, hours = _get(clean, "occupation"), _get(clean, "usual_hours_worked_gss")
    if occ is None and hours is not None and int(hours) > 0:
        resolved.append(f"nulled donor usual_hours_worked_gss={int(hours)} (ACS not in labor force)")
        clean["usual_hours_worked_gss"] = None
    return clean, resolved


# --- provenance ----------------------------------------------------------------

def build_provenance(
    row: Mapping[str, Any],
    *,
    backstory_model: str | None,
    reflection_model: str | None,
    seed: int,
    backstory_temperature: float,
    reflection_temperature: float,
) -> dict:
    """Per-persona R29 provenance: which real donor respondent each item came from +
    match distance, plus the per-stage generation params (each stage may use a
    different model, R6) so the persona is reproducible and traceable to the model
    that wrote each part."""
    donors: dict[str, dict] = {}
    for name, id_col, dist_col in _PROVENANCE_COLUMNS:
        rid, dist = _get(row, id_col), _get(row, dist_col)
        if rid is not None:
            donors[name] = {"respondent_id": rid, "match_distance": dist}
    return {
        "person_id": _get(row, "person_id"),
        "donors": donors,
        "generation": {
            "seed": seed,
            "backstory": {"model": backstory_model, "temperature": backstory_temperature, "prompt": BACKSTORY_PROMPT_V},
            "reflection": {"model": reflection_model, "temperature": reflection_temperature, "prompt": REFLECTION_PROMPT_V},
        },
    }


# --- the seeded persona --------------------------------------------------------

@dataclass(frozen=True)
class SeededPersona:
    """A generated persona ready to seed: the :class:`Persona` (with its reflection
    anchor + US location) plus its t=0 seed memories and generation provenance."""

    persona: Persona
    memories: tuple[SeedSpec, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        p = self.persona
        return {
            "persona": {
                "id": p.id,
                "description": p.description,
                "temperature": p.temperature,
                "values": list(p.values),
                "dispositions": list(p.dispositions),
                "location": p.location,
            },
            "memories": [{"text": m.text, "importance": m.importance, "created_at": m.created_at} for m in self.memories],
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SeededPersona":
        p = d["persona"]
        persona = Persona(
            id=p["id"],
            description=p["description"],
            temperature=p.get("temperature", 0.8),
            values=tuple(p.get("values", ())),
            dispositions=tuple(p.get("dispositions", ())),
            location=p.get("location", "New York City"),
        )
        memories = tuple(
            SeedSpec(text=m["text"], importance=m["importance"], created_at=m["created_at"])
            for m in d.get("memories", [])
        )
        return cls(persona=persona, memories=memories, provenance=dict(d.get("provenance", {})))


# --- the LLM steps -------------------------------------------------------------

def _complete(client: Any, system: str, user: str, *, temperature: float) -> str:
    return client.complete(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )


def _parse_reflection(text: str) -> tuple[list[str], list[str]] | None:
    """Parse the reflection step's JSON ``{"values": [...], "dispositions": [...]}``,
    tolerating stray prose or code fences around it."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    values = [str(x) for x in data.get("values", []) if str(x).strip()]
    dispositions = [str(x) for x in data.get("dispositions", []) if str(x).strip()]
    if not values and not dispositions:
        return None
    return values, dispositions


def _reflect(client: Any, lines: Sequence[str], *, temperature: float,
             label: str = "", attempts: int = 3) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reflection → anchor (Variant A): synthesize values/dispositions from the real
    measure lines **alone** (no demographics — see ``prompts.reflection_user``). Retries
    a soft-JSON parse; on persistent failure returns empty anchors (the persona still
    builds, just leaning on its seed memories) and warns."""
    user = prompts.reflection_user(lines)
    for _ in range(attempts):
        parsed = _parse_reflection(_complete(client, prompts.REFLECTION_SYSTEM, user, temperature=temperature))
        if parsed is not None:
            values, dispositions = parsed
            return tuple(values), tuple(dispositions)
    logger.warning("reflection produced no parseable anchor for %s; leaving anchor empty", label or "persona")
    return (), ()


def _model_of(client: Any) -> str | None:
    return getattr(getattr(client, "config", None), "model", None)


def build_persona(
    row: Mapping[str, Any],
    client: Any = None,
    *,
    backstory_client: Any = None,
    reflection_client: Any = None,
    seed: int = 0,
    backstory_temperature: float = 0.9,
    reflection_temperature: float = 0.3,
    persona_temperature: float = 0.8,
) -> SeededPersona:
    """Build one :class:`SeededPersona` from a joint-dataset row via the two LLM steps.

    **Stage-differentiated models (R6).** The two seed-time steps have different task
    shapes, so they take (optionally) different backends. Pass ``backstory_client`` and
    ``reflection_client`` to route each stage to its own model, or a single ``client``
    to use one for both. Recommended: a **larger model for the reflection** step (it
    abstracts real trait measures into a coherent behavioral posture and must emit
    clean JSON — the harder task), and the baseline (or a capable creative model) for
    the backstory. Seed-time generation is one-shot and cached, so a large reflection
    model adds no per-run cost.

    Temperatures are stage-appropriate too: the backstory runs warm (creative, 0.9);
    the reflection runs cool (0.3) for consistent, parseable synthesis. Each stage's
    model + temperature is logged in provenance. ``persona_temperature`` is the
    persona's *runtime* generation param (R1), distinct from these seed-time temps.

    A client needs only a ``complete(messages, *, temperature)`` method (the real
    :class:`~polis.llm.LLMClient`, or a stub in tests)."""
    bs_client = backstory_client or client
    rf_client = reflection_client or client
    if bs_client is None or rf_client is None:
        raise ValueError(
            "provide a client (or per-stage backstory_client / reflection_client)"
        )

    # Bake out cross-donor fusion artifacts first, so no downstream stage (backstory,
    # reflection, or memory) ever sees a contradictory measure (ADR 0016 data hygiene).
    row, resolved = sanitize_row(row)
    demographics = describe_demographics(row)
    location = location_of(row)
    lines = reflection_lines(row)
    pid = str(_get(row, "person_id") or f"usa_{seed}")

    backstory = _complete(
        bs_client, prompts.BACKSTORY_SYSTEM,
        prompts.backstory_user(demographics, location, lines),
        temperature=backstory_temperature,
    ).strip()
    # Reflection sees only the measures (no demographics — withholding beats
    # instructing-to-ignore, ADR 0016); label is for the failure log only.
    values, dispositions = _reflect(
        rf_client, lines, temperature=reflection_temperature, label=pid
    )

    persona = Persona(
        id=pid,
        description=demographics + ".",
        temperature=persona_temperature,
        values=values,
        dispositions=dispositions,
        location=location,
    )

    # Seed memories at t=0 (importance set at creation, recency anchored to sim start).
    memories: list[SeedSpec] = [
        SeedSpec(f"I'm {demographics}, living in {location}.", 5.0, 0.0),
    ]
    statements = statement_memories(row)
    if statements:
        memories.append(SeedSpec("A few things about me: " + " ".join(statements), 6.0, 0.0))
    if backstory:
        memories.append(SeedSpec(backstory, 6.0, 0.0))

    provenance = build_provenance(
        row,
        backstory_model=_model_of(bs_client),
        reflection_model=_model_of(rf_client),
        seed=seed,
        backstory_temperature=backstory_temperature,
        reflection_temperature=reflection_temperature,
    )
    # Data-hygiene provenance: what was baked out (cross-donor artifacts) vs what
    # remains flagged-but-preserved (within-donor real patterns the reflection reconciles).
    remaining = measure_contradictions(row)
    if resolved or remaining:
        logger.warning("persona %s data hygiene — resolved: %s | flagged: %s", pid, resolved, remaining)
    provenance["data_flags"] = {"resolved": resolved, "flagged": remaining}
    return SeededPersona(persona=persona, memories=tuple(memories), provenance=provenance)


def _rows(data: Any) -> list[Mapping[str, Any]]:
    """Accept a pandas DataFrame or an iterable of dict-like rows."""
    if hasattr(data, "to_dict"):
        return data.to_dict("records")
    return list(data)


def build_corpus(
    data: Any,
    client: Any = None,
    *,
    backstory_client: Any = None,
    reflection_client: Any = None,
    seed: int = 0,
    backstory_temperature: float = 0.9,
    reflection_temperature: float = 0.3,
    persona_temperature: float = 0.8,
) -> list[SeededPersona]:
    """Build the full persona corpus from the joint dataset (DataFrame or row dicts).

    Stage-differentiated backends (R6): pass ``backstory_client`` / ``reflection_client``
    to route each seed-time stage to its own model (recommended: a larger model for
    reflection), or a single ``client`` for both. See :func:`build_persona`."""
    return [
        build_persona(
            row, client,
            backstory_client=backstory_client, reflection_client=reflection_client,
            seed=seed, backstory_temperature=backstory_temperature,
            reflection_temperature=reflection_temperature,
            persona_temperature=persona_temperature,
        )
        for row in _rows(data)
    ]


# --- corpus artifact (versioned, cached) ---------------------------------------

def _content_hash(personas: Sequence[dict]) -> str:
    return hashlib.sha256(
        json.dumps(personas, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def corpus_to_dict(corpus: Sequence[SeededPersona], *, meta: Mapping[str, Any] | None = None) -> dict:
    """Serialize the corpus to a versioned artifact dict with a content hash (R17)."""
    personas = [sp.to_dict() for sp in corpus]
    return {
        "meta": {
            "corpus_version": CORPUS_VERSION,
            "n": len(personas),
            "content_hash": _content_hash(personas),
            **(dict(meta) if meta else {}),
        },
        "personas": personas,
    }


def corpus_from_dict(d: Mapping[str, Any]) -> list[SeededPersona]:
    return [SeededPersona.from_dict(p) for p in d["personas"]]


def save_corpus(corpus: Sequence[SeededPersona], path: str, *, meta: Mapping[str, Any] | None = None) -> dict:
    """Write the corpus artifact to ``path`` (JSON); returns the artifact dict."""
    artifact = corpus_to_dict(corpus, meta=meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    return artifact


def load_corpus(path: str) -> dict:
    """Read a corpus artifact dict from ``path``."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
