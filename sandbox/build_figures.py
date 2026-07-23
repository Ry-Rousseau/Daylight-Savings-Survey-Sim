"""Build the Phase-6 result figures (plotnine -> output/figures/). Reads the saved
experiment CSVs; each figure is guarded so a missing file just prints a note."""
import pandas as pd
from plotnine import (
    aes, coord_flip, element_text, facet_wrap, geom_col, geom_hline, geom_line,
    geom_point, geom_text, ggplot, labs, position_dodge, scale_color_manual,
    scale_x_discrete, scale_y_continuous, theme,
)

from polis import viz_theme as vz

CAL = "data/phase6_calibration_full.csv"


def _note(msg):
    print("  [skip]", msg)


# --- HEADLINE: % choosing Permanent DST by model, vs YouGov reality ------------
def fig_headline():
    df = pd.read_csv(CAL)
    q4 = df[(df.question == "Q4_permanent") & (df.option == "Permanent DST")].copy()
    yougov = float(q4[q4.model == "YouGov"].pct.iloc[0])
    m = q4[q4.model != "YouGov"].copy()
    order = ["32b", "qwenmax", "qwenmax_reason", "sonnet"]
    m["model"] = pd.Categorical(m.model, categories=order, ordered=True)
    m = m.sort_values("model")
    m["label"] = m.model.map(vz.MODEL_LABELS)
    m["label"] = pd.Categorical(m.label, categories=[vz.MODEL_LABELS[x] for x in order], ordered=True)
    p = (
        ggplot(m, aes("label", "pct"))
        + geom_col(fill=vz.STANCE_COLORS["Permanent DST"], width=0.68)
        + geom_hline(yintercept=yougov, linetype="dashed", color=vz.REFERENCE, size=0.7)
        + geom_text(aes(label="pct"), va="bottom", nudge_y=1.5, size=11, color=vz.INK)
        + geom_text(x=3.5, y=yougov + 2.5, label=f"YouGov (real): {yougov:.0f}%",
                    size=10, color=vz.REFERENCE, ha="right")
        + scale_y_continuous(limits=[0, 60], expand=(0, 0))
        + labs(title="Only Claude Sonnet reproduces real US DST opinion",
               subtitle="Share of 100 census-seeded personas choosing permanent daylight saving time,\n"
                        "queried individually (no interaction). Real YouGov 2023: 50%.",
               x="", y="% choosing permanent DST")
        + vz.theme_polis()
    )
    vz.save_fig(p, "headline_dst_by_model", width=8, height=5)
    print("  headline_dst_by_model.png")


# --- Full YouGov calibration (Q4 distribution per model) -----------------------
def fig_calibration_q4():
    df = pd.read_csv(CAL)
    q4 = df[df.question == "Q4_permanent"].copy()
    q4["model"] = pd.Categorical(q4.model.map(vz.MODEL_LABELS),
                                 categories=[vz.MODEL_LABELS[x] for x in vz.MODEL_ORDER], ordered=True)
    opt_order = ["Permanent DST", "Permanent Standard", "No preference", "Not sure"]
    q4["option"] = pd.Categorical(q4.option, categories=opt_order, ordered=True)
    p = (
        ggplot(q4, aes("model", "pct", fill="option"))
        + geom_col(position=position_dodge(width=0.8), width=0.72)
        + vz.fill_stance(name="Response", limits=opt_order)
        + labs(title="YouGov calibration: which time to make permanent?",
               subtitle="Real Americans favor permanent DST 50â€“31; every Qwen model inverts it to standard.\n"
                        "Sonnet-5 recovers the real split. (individual queries, n=100)",
               x="", y="% of responses")
        + vz.theme_polis()
    )
    vz.save_fig(p, "calibration_q4", width=9, height=5)
    print("  calibration_q4.png")


# --- Two-axis convergence: vote converges, voice survives (Sonnet) -------------
def fig_convergence():
    frames = []
    for path, model in [("data/phase6_scale_trajectory.csv", "Qwen3-32B"),
                        ("data/phase6_scale_sonnet_trajectory.csv", "Claude Sonnet-5")]:
        try:
            t = pd.read_csv(path)
        except FileNotFoundError:
            return _note(f"{path} missing")
        t = t[["tick", "dominant_share", "cluster_count"]].copy()
        t["model"] = model
        frames.append(t)
    d = pd.concat(frames)
    long = d.melt(id_vars=["tick", "model"], value_vars=["dominant_share", "cluster_count"],
                  var_name="metric", value_name="value")
    long["metric"] = long.metric.map({"dominant_share": "Vote: dominant share",
                                       "cluster_count": "Voice: distinct clusters"})
    cols = {"Qwen3-32B": "#D55E00", "Claude Sonnet-5": "#0072B2"}
    p = (
        ggplot(long, aes("tick", "value", color="model"))
        + geom_line(size=1.1) + geom_point(size=2.4)
        + facet_wrap("metric", scales="free_y")
        + scale_color_manual(values=cols, name="Decide model")
        + labs(title="With a good model the vote converges but the voice survives",
               x="tick", y="")
        + vz.theme_polis() + theme(legend_position="top")
    )
    vz.save_fig(p, "convergence_two_axis", width=9, height=4.5)
    print("  convergence_two_axis.png")


# --- Deliberation A/B: broadcast vs deliberate endpoint ------------------------
def fig_deliberation():
    try:
        d = pd.read_csv("data/phase6_deliberation.csv")
    except FileNotFoundError:
        return _note("deliberation csv missing")
    long = d.melt(id_vars="mode", value_vars=["DST", "STD", "SWITCH", "NOPREF"],
                  var_name="option", value_name="count")
    long["option"] = pd.Categorical(long.option, categories=["DST", "STD", "SWITCH", "NOPREF"], ordered=True)
    long["mode"] = pd.Categorical(long["mode"], categories=["broadcast", "deliberate"], ordered=True)
    p = (
        ggplot(long, aes("mode", "count", fill="option"))
        + geom_col(width=0.6)
        + vz.fill_stance(name="Endpoint vote")
        + labs(title="Removing vote-broadcasting reduces convergence",
               subtitle="Endpoint survey after 5 ticks. Broadcast â†’ 98% standard; deliberate (reasons only) â†’\n"
                        "86%, with the 'keep switching' minority surviving (12 vs 1).",
               x="discourse mode", y="agents")
        + vz.theme_polis()
    )
    vz.save_fig(p, "deliberation_ab", width=7, height=5)
    print("  deliberation_ab.png")


# --- Topology invariance: ring ~ small-world ----------------------------------
def fig_topology():
    frames = []
    for path, name in [("data/phase6_scale_sonnet_trajectory.csv", "Small-world"),
                       ("data/phase6_scale_sonnet_ring_trajectory.csv", "Ring lattice")]:
        try:
            t = pd.read_csv(path)
        except FileNotFoundError:
            return _note(f"{path} missing")
        t = t[["tick", "dominant_share"]].copy(); t["topology"] = name
        frames.append(t)
    d = pd.concat(frames)
    p = (
        ggplot(d, aes("tick", "dominant_share", color="topology"))
        + geom_line(size=1.1) + geom_point(size=2.4)
        + scale_color_manual(values={"Small-world": "#0072B2", "Ring lattice": "#009E73"}, name="Topology")
        + scale_y_continuous(limits=[0.6, 1.0])
        + labs(title="Topology alone doesn't stop the collapse",
               subtitle="Sonnet, dominant vote share over ticks. Sparse ring â‰ˆ small-world â€” the standard-time\n"
                        "lean is uniform, so every neighborhood converges the same way.",
               x="tick", y="dominant vote share")
        + vz.theme_polis() + theme(legend_position="top")
    )
    vz.save_fig(p, "topology_invariance", width=8, height=4.5)
    print("  topology_invariance.png")


# --- Opinion/conviction matrix ------------------------------------------------
def fig_opinion():
    try:
        d = pd.read_csv("data/phase6_opinion_experiments.csv")
    except FileNotFoundError:
        return _note("opinion experiments csv missing")
    d = d.sort_values("end_dom_share")
    d["experiment"] = pd.Categorical(d.experiment, categories=list(d.experiment), ordered=True)
    p = (
        ggplot(d, aes("experiment", "end_dom_share"))
        + geom_col(fill="#0072B2", width=0.68)
        + geom_text(aes(label="end_dom_share"), va="center", nudge_y=-0.05, color="white", size=10)
        + coord_flip()
        + scale_y_continuous(limits=[0, 1.05], expand=(0, 0))
        + labs(title="Only committed minorities preserve a vote split",
               subtitle="Endpoint dominant share by seeding condition (lower = more split).",
               x="", y="endpoint dominant share")
        + vz.theme_polis()
    )
    vz.save_fig(p, "opinion_matrix", width=8, height=5)
    print("  opinion_matrix.png")


if __name__ == "__main__":
    print("building figures -> output/figures/")
    for fn in (fig_headline, fig_calibration_q4, fig_convergence, fig_deliberation,
               fig_topology, fig_opinion):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - one bad figure shouldn't abort the rest
            print(f"  [error] {fn.__name__}: {e!r}")
    print("done.")

