"""Claude Sonnet run — stance breakdown across ticks. Extracts the per-tick stance
distribution from the run log and plots it (stacked area + 100% stacked bar)."""
import json
import sqlite3
from collections import Counter

import pandas as pd
from plotnine import (
    aes, element_text, geom_area, geom_bar, geom_col, geom_text, ggplot, labs,
    position_fill, position_stack, scale_x_continuous, scale_y_continuous, theme,
)

from polis import viz_theme as vz

SHORT = {"Adopt permanent daylight saving time (clocks stay on summer time all year: later sunrises, later sunsets)": "DST",
         "Adopt permanent standard time (clocks stay on winter time all year: earlier sunrises, earlier sunsets)": "STD",
         "Keep switching the clocks twice a year": "SWITCH",
         "No preference": "NOPREF"}
TICKS = 5

c = sqlite3.connect("data/phase6_scale_sonnet.sqlite")
speaks = []
for aid, t, pj in c.execute("SELECT agent_id, tick, payload_json FROM events WHERE event_type='action' ORDER BY event_id"):
    p = json.loads(pj)
    if p.get("action_type") == "speak" and p.get("stance"):
        speaks.append((t, aid, SHORT.get(p["stance"], p["stance"][:8])))

rows = []
for T in range(TICKS):
    latest = {}
    for t, aid, stance in speaks:
        if t <= T:
            latest[aid] = stance  # append order -> later tick overwrites (as-of-T)
    for stance, n in Counter(latest.values()).items():
        rows.append({"tick": T, "stance": stance, "count": n})
df = pd.DataFrame(rows)
df.to_csv("data/phase6_sonnet_stance_trajectory.csv", index=False)
print(df.pivot(index="tick", columns="stance", values="count").fillna(0).astype(int).to_string())

ORDER = ["STD", "SWITCH", "DST", "NOPREF"]  # bottom -> top; the growing STD sits at base
df["stance"] = pd.Categorical(df.stance, categories=ORDER, ordered=True)

# stacked-area flow (counts)
p = (
    ggplot(df, aes("tick", "count", fill="stance"))
    + geom_area(position=position_stack(reverse=True), color="white", size=0.4)
    + vz.fill_stance(name="Stance")
    + scale_x_continuous(breaks=list(range(TICKS)), expand=(0, 0))
    + scale_y_continuous(expand=(0, 0))
    + labs(title="Claude Sonnet-5: stance breakdown converges over the run",
           subtitle="100 census-seeded personas, small-world graph. The tick-0 split (standard / keep-switching)\n"
                    "collapses toward permanent standard time as agents hear each other.",
           x="tick", y="number of agents")
    + vz.theme_polis()
)
vz.save_fig(p, "sonnet_stance_trajectory", width=8.5, height=5)
print("  sonnet_stance_trajectory.png")

# 100% stacked bars (share) — the same story as proportions
p = (
    ggplot(df, aes("factor(tick)", "count", fill="stance"))
    + geom_col(position=position_fill(reverse=True), width=0.82)
    + vz.fill_stance(name="Stance")
    + scale_y_continuous(labels=lambda l: [f"{int(x*100)}%" for x in l], expand=(0, 0))
    + labs(title="Claude Sonnet-5: stance share by tick",
           subtitle="Share of the 100 personas on each stance, tick 0 -> 4.",
           x="tick", y="share of agents")
    + vz.theme_polis()
)
vz.save_fig(p, "sonnet_stance_share_by_tick", width=8, height=5)
print("  sonnet_stance_share_by_tick.png")
