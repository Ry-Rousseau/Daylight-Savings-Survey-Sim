"""Build the presentation comparison for the base-persona tick-0 survey."""
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    geom_col,
    ggplot,
    labs,
    position_dodge,
    scale_fill_manual,
    scale_y_continuous,
    theme,
)

from polis import viz_theme as vz


DATA = "data/phase6_calibration_full.csv"
QUESTION = "Q4_permanent"
SOURCE_ORDER = ["YouGov", "sonnet"]
SOURCE_LABELS = {
    "YouGov": "YouGov",
    "sonnet": "Claude base personas",
}
SOURCE_COLORS = {
    "YouGov": vz.STANCE_COLORS["Permanent Standard"],
    "Claude base personas": vz.STANCE_COLORS["Permanent DST"],
}
OPTION_ORDER = [
    "Permanent DST",
    "Permanent Standard",
    "No preference",
    "Not sure",
]
OPTION_LABELS = {
    "Permanent DST": "Permanent daylight\nsaving time",
    "Permanent Standard": "Permanent\nstandard time",
    "No preference": "No preference",
    "Not sure": "Not sure",
}


df = pd.read_csv(DATA)
plot_df = df[
    (df["question"] == QUESTION) & df["model"].isin(SOURCE_ORDER)
].copy()
plot_df["source"] = plot_df["model"].map(SOURCE_LABELS)
plot_df["source"] = pd.Categorical(
    plot_df["source"],
    categories=[SOURCE_LABELS[source] for source in SOURCE_ORDER],
    ordered=True,
)
plot_df["response"] = plot_df["option"].map(OPTION_LABELS)
plot_df["response"] = pd.Categorical(
    plot_df["response"],
    categories=[OPTION_LABELS[option] for option in OPTION_ORDER],
    ordered=True,
)
plot_df["value_label"] = plot_df["pct"].astype(int).astype(str) + "%"

chart = (
    ggplot(plot_df, aes("response", "pct", fill="source"))
    + geom_col(position=position_dodge(width=0.76), width=0.66)
    + scale_fill_manual(values=SOURCE_COLORS, name="")
    + scale_y_continuous(
        limits=[0, 60],
        breaks=[0, 20, 40, 60],
        labels=lambda values: [f"{int(value)}%" for value in values],
        expand=(0, 0),
    )
    + labs(
        title="Claude comes within 5 points of YouGov on permanent DST",
        x="",
        y="Response share",
    )
    + vz.theme_polis(base_size=14)
    + theme(
        figure_size=(10, 5.6),
        legend_position="top",
        legend_justification="left",
        axis_text_x=element_text(ha="center"),
        panel_grid_major_x=element_blank(),
    )
)

path = vz.save_fig(
    chart,
    "tick0_yougov_vs_sonnet",
    width=10,
    height=5.6,
)
print(path)



