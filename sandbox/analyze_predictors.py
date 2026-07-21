"""What persona attribute predicts DST stance? Cramér's V per demographic + breakdown
figures. Reads data/phase6_persona_stances.csv (per-persona Sonnet answers + demographics)."""
import pandas as pd
from plotnine import (
    aes, coord_flip, geom_bar, geom_col, geom_text, ggplot, labs,
    scale_y_continuous, theme, element_text,
)
from scipy.stats import chi2_contingency

from polis import viz_theme as vz

d = pd.read_csv("data/phase6_persona_stances.csv")
d = d[d.stance.isin(["DST", "Standard"])].copy()  # binary contrast (drop the 1 no-pref)
d["stance"] = d.stance.map({"DST": "DST", "Standard": "STD"})

# --- derived / binned features -------------------------------------------------
d["age_band"] = pd.cut(d.age, [17, 34, 49, 64, 200], labels=["18-34", "35-49", "50-64", "65+"])
wt = pd.to_datetime(d.wake_time_atus, format="%H:%M:%S", errors="coerce").dt.hour
d["chronotype"] = pd.cut(wt, [-1, 6, 8, 24], labels=["Early (<7am)", "Mid (7-9am)", "Late (9am+)"])
d["employment"] = d.occupation.apply(lambda s: "Not working" if str(s).startswith("N/A") else "Working")
d["has_kids"] = d.presence_of_children.astype(str).str.contains("With own children", case=False).map(
    {True: "Kids at home", False: "No kids at home"})


def edu_band(s):
    s = str(s).lower()
    if any(k in s for k in ("master", "doctor", "professional")):
        return "Graduate"
    if "bachelor" in s:
        return "Bachelor's"
    if any(k in s for k in ("college", "associate")):
        return "Some college"
    return "HS or less"


def race_band(s):
    s = str(s)
    for k, v in [("White", "White"), ("Black", "Black"), ("Hispanic", "Hispanic")]:
        if k in s:
            return v
    return "Other"


# Bin the high-cardinality features so Cramer's V is comparable across features
# (16 education levels / 7 race levels on n=99 otherwise inflate the statistic).
d["education_band"] = d.education.apply(edu_band)
d["race_band"] = d.race_ethnicity.apply(race_band)

FEATURES = {"age_band": "Age", "education_band": "Education", "race_band": "Race/ethnicity",
            "region": "Region", "chronotype": "Chronotype (wake time)", "sex": "Sex",
            "employment": "Employment", "has_kids": "Children at home"}


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    if min(ct.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    return float((chi2 / (n * (min(ct.shape) - 1))) ** 0.5)


rank = pd.DataFrame(
    [{"col": col, "feature": lbl, "cramers_v": round(cramers_v(d[col], d.stance), 3)}
     for col, lbl in FEATURES.items()]
).sort_values("cramers_v", ascending=False)
print("=== strongest predictor of DST vs Standard (Cramer's V) ===")
print(rank[["feature", "cramers_v"]].to_string(index=False))
print(f"\noverall: DST {int((d.stance=='DST').sum())} / STD {int((d.stance=='STD').sum())}")
for col, lbl in [(r.col, r.feature) for r in rank.head(3).itertuples()]:
    print(f"\n{lbl} — DST rate by category:")
    print(d.groupby(col, observed=True).stance.apply(lambda s: round((s == "DST").mean(), 2)).to_string())


# --- figure: predictor ranking -------------------------------------------------
rk = rank.copy()
rk["feature"] = pd.Categorical(rk.feature, categories=list(rk.feature[::-1]), ordered=True)
p = (
    ggplot(rk, aes("feature", "cramers_v"))
    + geom_col(fill="#0072B2", width=0.66)
    + geom_text(aes(label="cramers_v"), va="center", nudge_y=0.012, size=10, color=vz.INK)
    + coord_flip()
    + labs(title="What predicts a persona's DST stance?",
           subtitle="Cramer's V with the DST-vs-standard choice (n=99). All weak; age is the cleanest.",
           x="", y="Cramer's V  (0 = none, 1 = perfect)")
    + vz.theme_polis()
)
vz.save_fig(p, "predictor_ranking", width=8, height=4.5)
print("\n  predictor_ranking.png")

# --- figures: stance share by the demographics of interest ---------------------
for col, lbl in [("age_band", "Age"), ("region", "Region"), ("education_band", "Education")]:
    sub = d.dropna(subset=[col]).copy()
    sub[col] = sub[col].astype(str)
    sub["stance"] = pd.Categorical(sub.stance, categories=["DST", "STD"], ordered=True)
    p = (
        ggplot(sub, aes(col, fill="stance"))
        + geom_bar(position="fill", width=0.7)
        + vz.fill_stance(name="Stance")
        + scale_y_continuous(labels=lambda l: [f"{int(x*100)}%" for x in l], expand=(0, 0))
        + labs(title=f"DST stance by {lbl.lower()}",
               subtitle="Share of personas choosing permanent DST vs standard.",
               x=lbl, y="share of personas")
        + vz.theme_polis() + theme(axis_text_x=element_text(rotation=15, ha="right"))
    )
    vz.save_fig(p, f"stance_by_{col}", width=7.5, height=4.5)
    print(f"  stance_by_{col}.png")
