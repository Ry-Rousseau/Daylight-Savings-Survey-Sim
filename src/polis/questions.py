"""Canonical survey questions (Layer 5 — Interface/Query).

One definition, imported by both the `python -m polis` demo and the notebooks,
so the question text never drifts between them.

DST wording note: an 8B model reliably reasons about "earlier/later sunrises"
but does *not* reliably know that "permanent daylight saving time" means later
sunrises and "permanent standard time" means earlier ones — the same confusion
that makes real DST polls wording-sensitive. The options therefore spell out the
sunrise/sunset consequence so a memory-driven preference maps to the right
option instead of being lost to a labeling gap. This is clarification, not a
thumb on the scale: with no seeded memory the model picks neither pole.
"""
from __future__ import annotations

from .survey import SurveyQuestion

# Index 0 is the permanent-DST option tracked as P(permanent-DST) in the DoD.
DST_OPTIONS = [
    "Adopt permanent daylight saving time (clocks stay on summer time all year: later sunrises, later sunsets)",
    "Adopt permanent standard time (clocks stay on winter time all year: earlier sunrises, earlier sunsets)",
    "Keep switching the clocks twice a year",
    "No preference",
]

DST_QUESTION = SurveyQuestion(
    text="What should the United States do about daylight saving time?",
    options=DST_OPTIONS,
)

DST_PERMANENT = DST_OPTIONS[0]  # the tracked target option

# Un-annotated variant: bare option names, no sunrise/sunset gloss. Same order
# and count as DST_OPTIONS so answers are comparable by option index. Used by the
# model-capability sweep (ADR 0005) to test whether a larger model maps
# "permanent DST" -> later sunrises unaided, i.e. whether the annotation is a
# crutch only the 8B needs.
DST_OPTIONS_PLAIN = [
    "Adopt permanent daylight saving time",
    "Adopt permanent standard time",
    "Keep switching the clocks twice a year",
    "No preference",
]

DST_QUESTION_PLAIN = SurveyQuestion(
    text="What should the United States do about daylight saving time?",
    options=DST_OPTIONS_PLAIN,
)

# --- YouGov calibration questions (reference/survey_validation) -----------------
# Verbatim wording from the YouGov "Daylight Saving Time" survey (1000 U.S. adults,
# March 6-9 2023), so the silicon sample is asked exactly what real respondents were
# — the calibration target. Real toplines are recorded here as the ground truth to
# score against. Q4 is de-conditionalised (the original was asked only of those who
# said "eliminate" in Q2, prefaced "You indicated…"); we drop that preface so it can be
# asked of everyone as a standalone item.

YOUGOV_Q2_ELIMINATE = SurveyQuestion(
    text=("Would you like to see the changing of the clocks eliminated, so people no "
          "longer change their clocks twice per year?"),
    options=["Yes, I would", "No, I would not", "Not sure"],
)
# YouGov topline: Yes 62% · No 21% · Not sure 17%
YOUGOV_Q2_TOPLINE = {"Yes, I would": 0.62, "No, I would not": 0.21, "Not sure": 0.17}

YOUGOV_Q4_PERMANENT = SurveyQuestion(
    text="Which time would you like to make permanent?",
    options=[
        "Permanent Daylight Saving Time (later sunrises and sunsets)",
        "Permanent Standard Time (earlier sunrises and sunsets)",
        "No preference",
        "Not sure",
    ],
)
# YouGov topline (asked of eliminators): Permanent DST 50% · Permanent Standard 31% ·
# No preference 12% · Not sure 7%.
YOUGOV_Q4_TOPLINE = {
    "Permanent Daylight Saving Time (later sunrises and sunsets)": 0.50,
    "Permanent Standard Time (earlier sunrises and sunsets)": 0.31,
    "No preference": 0.12,
    "Not sure": 0.07,
}
