"""Silicon sample vs YouGov, by demographic cross-section (permanent-DST %).
YouGov Q4 crosstab values transcribed from reference/survey_validation/*.pdf (pp. 7-8);
our numbers computed from the per-persona Sonnet survey."""
import pandas as pd
from plotnine import (
    aes, element_text, facet_wrap, geom_col, geom_hline, geom_text, ggplot, labs,
    position_dodge, scale_fill_manual, scale_y_continuous, theme,
)

from polis import viz_theme as vz

# --- YouGov Q4 "% permanent DST" crosstab (real ground truth) ------------------
YOUGOV = {
    "Age": {"18-29": 45, "30-44": 43, "45-64": 54, "65+": 52},
    "Sex": {"Male": 45, "Female": 54},
    "Region": {"Northeast": 53, "Midwest": 52, "South": 49, "West": 45},
}
YOUGOV_TOTAL = 50

# --- our per-persona Sonnet answers -> % choosing permanent DST ----------------
d = pd.read_csv("data/phase6_persona_stances.csv")
d["is_dst"] = (d.stance == "DST").astype(int)
d["Age"] = pd.cut(d.age, [17, 29, 44, 64, 200], labels=["18-29", "30-44", "45-64", "65+"])
d["Sex"] = d.sex
d["Region"] = d.region

rows = []
for demo in ("Age", "Sex", "Region"):
    ours = d.groupby(demo, observed=True).is_dst.mean().mul(100).round(0)
    for cat, yg in YOUGOV[demo].items():
        rows.append({"demo": demo, "category": str(cat), "source": "YouGov (real)", "dst": yg})
        rows.append({"demo": demo, "category": str(cat), "source": "Silicon (Sonnet)",
                     "dst": float(ours.get(cat, float("nan")))})
c = pd.DataFrame(rows).dropna()
c["source"] = pd.Categorical(c.source, categories=["YouGov (real)", "Silicon (Sonnet)"], ordered=True)
SRC = {"YouGov (real)": "#D55E00", "Silicon (Sonnet)": "#0072B2"}

print(c.pivot_table(index=["demo", "category"], columns="source", values="dst").to_string())

# --- faceted comparison across all three cross-sections ------------------------
p = (
    ggplot(c, aes("category", "dst", fill="source"))
    + geom_col(position=position_dodge(width=0.78), width=0.7)
    + geom_hline(yintercept=YOUGOV_TOTAL, linetype="dotted", color=vz.MUTED, size=0.5)
    + facet_wrap("demo", scales="free_x")
    + scale_fill_manual(values=SRC, name="")
    + scale_y_continuous(limits=[0, 70], expand=(0, 0))
    + labs(title="Silicon sample vs YouGov: % choosing permanent DST, by demographic",
           subtitle="Aggregate level is close (~40 vs 50), but the age gradient inverts: our model has\n"
                    "younger personas favoring DST; real Americans show the opposite.",
           x="", y="% permanent DST")
    + vz.theme_polis() + theme(legend_position="top", axis_text_x=element_text(rotation=20, ha="right"))
)
vz.save_fig(p, "compare_vs_yougov_by_demo", width=11, height=4.8)
print("  compare_vs_yougov_by_demo.png")

# --- standalone: the age comparison (the headline divergence) ------------------
age = c[c.demo == "Age"].copy()
p = (
    ggplot(age, aes("category", "dst", fill="source"))
    + geom_col(position=position_dodge(width=0.78), width=0.68)
    + geom_text(aes(label="dst.astype(int)"), position=position_dodge(width=0.78),
                va="bottom", nudge_y=1, size=10, color=vz.INK)
    + scale_fill_manual(values=SRC, name="")
    + scale_y_continuous(limits=[0, 70], expand=(0, 0))
    + labs(title="The age gradient is inverted",
           subtitle="% choosing permanent DST by age. Real Americans: flat/rising with age.\n"
                    "Our Sonnet personas: falling with age — a stereotype the data doesn't support.",
           x="age", y="% permanent DST")
    + vz.theme_polis() + theme(legend_position="top")
)
vz.save_fig(p, "compare_vs_yougov_age", width=8, height=5)
print("  compare_vs_yougov_age.png")
