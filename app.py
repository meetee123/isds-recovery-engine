"""
ISDS Recovery Realism Engine — Streamlit Application
=====================================================
Professional dashboard for investor-state dispute settlement analysis,
Monte Carlo simulation, and enforcement pathway mapping for African states.

Run with:
    streamlit run app.py
from the isds_app/ directory.
"""

from __future__ import annotations

import sys
import os

# Ensure the isds_app directory is on the path when run from elsewhere
sys.path.insert(0, os.path.dirname(__file__))

import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data_module import (
    CASES,
    COUNTRY_PROFILES,
    ENFORCEMENT_JURISDICTIONS,
    SECTOR_STATS,
    TREATY_BASIS_STATS,
    calculate_historical_rates,
    get_cases_by_country,
)
from simulation_engine import (
    BehavioralModule,
    DisputeProfile,
    EnforcementPathway,
    SimulationEngine,
)
from memo_generator import MemoGenerator

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Investor-State Dispute Enforcement Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------

PRIMARY    = "#20808D"
SECONDARY  = "#13343B"
ACCENT     = "#A84B2F"
BG         = "#FCFAF6"
CARD_BG    = "#F0F4F4"
TEXT_DARK  = "#1A2B2E"
TEXT_LIGHT = "#FFFFFF"

st.markdown(f"""
<style>
    /* Root background */
    .stApp {{
        background-color: {BG};
    }}
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {SECONDARY};
        color: {TEXT_LIGHT};
    }}
    [data-testid="stSidebar"] * {{
        color: {TEXT_LIGHT} !important;
    }}
    [data-testid="stSidebar"] a {{
        color: #A8D8DC !important;
    }}
    /* Metric cards */
    [data-testid="stMetricValue"] {{
        font-size: 1.6rem !important;
        color: {PRIMARY} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.78rem !important;
        color: {SECONDARY} !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 0.82rem !important;
    }}
    /* Tab headers */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {SECONDARY};
        border-radius: 8px 8px 0 0;
        padding: 4px 8px 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: rgba(255,255,255,0.65) !important;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY} !important;
        color: {TEXT_LIGHT} !important;
    }}
    /* Section headers */
    .section-header {{
        background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
        color: {TEXT_LIGHT};
        padding: 10px 18px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.05rem;
        margin: 18px 0 10px 0;
        letter-spacing: 0.04em;
    }}
    /* Info cards */
    .info-card {{
        background-color: {CARD_BG};
        border-left: 4px solid {PRIMARY};
        padding: 14px 18px;
        border-radius: 0 6px 6px 0;
        margin: 8px 0;
        font-size: 0.9rem;
        color: {TEXT_DARK};
    }}
    /* Risk cards */
    .risk-critical {{ border-left-color: {ACCENT} !important; background-color: #FFF0ED !important; }}
    .risk-high     {{ border-left-color: #E07B39 !important; background-color: #FFF7F0 !important; }}
    .risk-moderate {{ border-left-color: #E0B839 !important; background-color: #FDFAF0 !important; }}
    .risk-low      {{ border-left-color: #4CAF7D !important; background-color: #F0FBF5 !important; }}
    /* Buttons */
    .stButton > button {{
        background-color: {PRIMARY};
        color: {TEXT_LIGHT};
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: background 0.2s;
    }}
    .stButton > button:hover {{
        background-color: {SECONDARY};
    }}
    /* Download buttons */
    .stDownloadButton > button {{
        background-color: {ACCENT};
        color: {TEXT_LIGHT};
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }}
    /* DataFrames */
    [data-testid="stDataFrame"] {{
        border: 1px solid #D0E4E6;
        border-radius: 6px;
    }}
    /* Expander */
    [data-testid="stExpander"] {{
        border: 1px solid #D0E4E6;
        border-radius: 6px;
    }}
    /* Footer */
    .app-footer {{
        text-align: center;
        color: #888;
        font-size: 0.75rem;
        padding: 20px 0 10px;
        border-top: 1px solid #D0E4E6;
        margin-top: 30px;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def fmt_usd(amount: Optional[float], default: str = "N/A") -> str:
    if amount is None:
        return default
    if abs(amount) >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f}B"
    if abs(amount) >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    return f"${amount:,.0f}"


def fmt_pct(v: Optional[float], default: str = "N/A") -> str:
    if v is None:
        return default
    return f"{v:.1%}"


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def info_card(text: str, risk_level: str = "") -> None:
    cls = "info-card"
    if risk_level:
        cls += f" risk-{risk_level.lower()}"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_DARK),
    colorway=[PRIMARY, ACCENT, "#4CAF7D", "#E0B839", "#7B61FF", "#E07B39"],
)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

@st.cache_data
def load_cases_df() -> pd.DataFrame:
    rows = []
    for c in CASES:
        rows.append({
            "Case Name": c["case_name"],
            "Country": c["respondent_state"],
            "Investor Nationality": c["investor_nationality"],
            "Sector": c["sector"],
            "Treaty Basis": c["treaty_basis"],
            "Year Filed": c["year_filed"],
            "Year Decided": c["year_decided"],
            "Outcome": c["outcome"],
            "Amount Claimed ($M)": (c["amount_claimed_usd"] / 1e6) if c["amount_claimed_usd"] else None,
            "Amount Awarded ($M)": (c["amount_awarded_usd"] / 1e6) if c["amount_awarded_usd"] else None,
            "Annulment Attempted": c["annulment_attempted"],
            "Annulment Outcome": c["annulment_outcome"] or "N/A",
            "Enforcement Status": c["enforcement_status"],
            "Notes": c["notes"],
        })
    return pd.DataFrame(rows)


@st.cache_data
def get_summary_stats(df: pd.DataFrame) -> dict:
    resolved = df[~df["Outcome"].isin(["Pending", "Discontinued"])]
    investor_wins = (resolved["Outcome"] == "Investor Win").sum()
    n_resolved = len(resolved)
    win_rate = investor_wins / n_resolved if n_resolved > 0 else 0.0

    both_known = df.dropna(subset=["Amount Claimed ($M)", "Amount Awarded ($M)"])
    both_known = both_known[
        (both_known["Amount Claimed ($M)"] > 0) & (both_known["Amount Awarded ($M)"] > 0)
    ]
    avg_ratio = (
        (both_known["Amount Awarded ($M)"] / both_known["Amount Claimed ($M)"]).mean()
        if len(both_known) > 0 else None
    )

    return {
        "total_cases": len(df),
        "investor_win_rate": win_rate,
        "avg_award_to_claim": avg_ratio,
        "total_claimed": df["Amount Claimed ($M)"].sum(),
        "total_awarded": df["Amount Awarded ($M)"].sum(),
    }


# ---------------------------------------------------------------------------
# CHART FACTORIES
# ---------------------------------------------------------------------------

def make_bar_cases_by_country(df: pd.DataFrame) -> go.Figure:
    counts = df["Country"].value_counts().reset_index()
    counts.columns = ["Country", "Count"]
    fig = px.bar(
        counts.sort_values("Count"),
        x="Count", y="Country", orientation="h",
        color="Count",
        color_continuous_scale=[[0, CARD_BG], [1, PRIMARY]],
        labels={"Count": "Number of Cases", "Country": ""},
        title="Cases by Respondent State",
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        coloraxis_showscale=False,
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_donut_outcomes(df: pd.DataFrame) -> go.Figure:
    counts = df["Outcome"].value_counts()
    colours = {
        "Investor Win": PRIMARY,
        "State Win": ACCENT,
        "Settled": "#4CAF7D",
        "Pending": "#E0B839",
        "Discontinued": "#AAAAAA",
        "Annulled": "#7B61FF",
    }
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker_colors=[colours.get(o, "#CCCCCC") for o in counts.index],
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title="Case Outcomes Distribution",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="v", x=1.02),
    )
    return fig


def make_timeline_filings(df: pd.DataFrame) -> go.Figure:
    yr = df.dropna(subset=["Year Filed"]).copy()
    yr["Year Filed"] = yr["Year Filed"].astype(int)
    by_year = yr.groupby("Year Filed").size().reset_index(name="Count")
    fig = px.area(
        by_year, x="Year Filed", y="Count",
        title="Cases Filed Over Time",
        labels={"Year Filed": "Year", "Count": "Cases Filed"},
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_traces(fill="tozeroy", fillcolor=f"rgba(32,128,141,0.15)", line_color=PRIMARY)
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(dtick=5),
    )
    return fig


def make_recovery_histogram(recovery_dist: np.ndarray) -> go.Figure:
    mean_r = float(recovery_dist.mean())
    p25 = float(np.percentile(recovery_dist, 25))
    p75 = float(np.percentile(recovery_dist, 75))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=recovery_dist * 100,
        nbinsx=60,
        name="Simulated Recovery Rate",
        marker_color=PRIMARY,
        opacity=0.8,
        hovertemplate="Recovery: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))
    # Vertical lines
    for val, label, colour in [
        (mean_r * 100, f"Mean {mean_r:.1%}", PRIMARY),
        (p25 * 100, f"P25 {p25:.1%}", "#4CAF7D"),
        (p75 * 100, f"P75 {p75:.1%}", ACCENT),
    ]:
        fig.add_vline(x=val, line_dash="dash", line_color=colour, line_width=2,
                      annotation_text=label, annotation_position="top",
                      annotation_font_color=colour)

    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title="Simulated Recovery Rate Distribution (% of Claim)",
        xaxis_title="Recovery Rate (% of Claim)",
        yaxis_title="Simulation Count",
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        showlegend=False,
    )
    return fig


def make_award_histogram(award_dist: np.ndarray, claimed: float) -> go.Figure:
    awards_m = award_dist * claimed / 1e6
    mean_a = float(awards_m.mean())
    median_a = float(np.median(awards_m))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=awards_m,
        nbinsx=60,
        name="Simulated Award ($M)",
        marker_color=ACCENT,
        opacity=0.8,
        hovertemplate="Award: $%{x:.1f}M<br>Count: %{y}<extra></extra>",
    ))
    for val, label, colour in [
        (mean_a, f"Mean ${mean_a:.1f}M", ACCENT),
        (median_a, f"Median ${median_a:.1f}M", PRIMARY),
    ]:
        fig.add_vline(x=val, line_dash="dash", line_color=colour, line_width=2,
                      annotation_text=label, annotation_position="top",
                      annotation_font_color=colour)

    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title="Simulated Award Amount Distribution ($M)",
        xaxis_title="Award Amount ($M)",
        yaxis_title="Simulation Count",
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        showlegend=False,
    )
    return fig


def make_timeline_box(timeline_dist: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=timeline_dist / 12,  # convert months → years
        box_visible=True,
        meanline_visible=True,
        fillcolor=f"rgba(32,128,141,0.3)",
        line_color=PRIMARY,
        name="Timeline (Years)",
        hovertemplate="Years: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title="Enforcement Timeline Distribution",
        yaxis_title="Years to Recovery",
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        showlegend=False,
    )
    return fig


def make_radar_friction(country: str) -> go.Figure:
    profile = COUNTRY_PROFILES.get(country, {})
    if not profile:
        return go.Figure()

    rol = profile.get("wgi_rule_of_law", 50.0)
    cor = profile.get("wgi_corruption", 50.0)
    gov = profile.get("wgi_govt_effectiveness", 50.0)
    wjp = (profile.get("wjp_score") or 0.5) * 100
    comp_map = {"Yes": 90.0, "Partial": 55.0, "No": 10.0, "Unknown": 45.0}
    comp = comp_map.get(profile.get("voluntary_compliance_history", "Unknown"), 45.0)

    categories = ["Rule of Law", "Corruption Control", "Govt Effectiveness", "WJP Score", "Compliance"]
    values = [rol, cor, gov, wjp, comp]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor=f"rgba(32,128,141,0.25)",
        line_color=PRIMARY,
        name=country,
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix=""),
            bgcolor=CARD_BG,
        ),
        title=f"Sovereign Friction Components — {country}",
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=False,
    )
    return fig


def make_settlement_bar(investor_floor: float, state_ceiling: float, award: float) -> go.Figure:
    fig = go.Figure()
    # Background: full award range
    fig.add_trace(go.Bar(
        y=["Settlement Zone"],
        x=[award],
        orientation="h",
        marker_color="rgba(200,200,200,0.3)",
        showlegend=False,
        hoverinfo="skip",
        base=0,
        name="Award (Face)",
    ))
    # ZOPA range
    zopa_width = max(0, state_ceiling - investor_floor)
    if zopa_width > 0:
        fig.add_trace(go.Bar(
            y=["Settlement Zone"],
            x=[zopa_width],
            base=[investor_floor],
            orientation="h",
            marker_color="rgba(76,175,125,0.5)",
            marker_line_color="#4CAF7D",
            marker_line_width=2,
            name=f"ZOPA ({fmt_usd(investor_floor)} – {fmt_usd(state_ceiling)})",
            hovertemplate="ZOPA: %{x:$.0f}<extra></extra>",
        ))
    # Investor floor line
    fig.add_vline(x=investor_floor, line_dash="solid", line_color="#4CAF7D", line_width=2.5,
                  annotation_text=f"Investor Floor<br>{fmt_usd(investor_floor)}",
                  annotation_position="top right", annotation_font_color="#4CAF7D")
    # State ceiling line
    fig.add_vline(x=state_ceiling, line_dash="solid", line_color=ACCENT, line_width=2.5,
                  annotation_text=f"State Ceiling<br>{fmt_usd(state_ceiling)}",
                  annotation_position="top left", annotation_font_color=ACCENT)
    # Award face value
    fig.add_vline(x=award, line_dash="dot", line_color=TEXT_DARK, line_width=1.5,
                  annotation_text=f"Award<br>{fmt_usd(award)}",
                  annotation_position="top left", annotation_font_color=TEXT_DARK)

    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title="Settlement Zone (ZOPA) Analysis",
        xaxis_title="USD",
        height=250,
        margin=dict(l=10, r=10, t=60, b=40),
        barmode="overlay",
        xaxis=dict(tickformat="$,.0f"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.35),
    )
    return fig


def make_prospect_theory_chart(
    award: float, inv_floor: float, state_ceil: float
) -> go.Figure:
    """Visualise prospect theory valuations across a range of settlement amounts."""
    bm = BehavioralModule()
    # Settlement amounts from 0 to 1.1× award
    amounts = np.linspace(0, award * 1.05, 200)

    inv_values, state_values = [], []
    for amt in amounts:
        pv_inv = bm.prospect_theory_valuation(amt, reference_point=award)
        # State reference = midpoint of their expected discount
        state_ref = (inv_floor + state_ceil) / 2 if state_ceil > 0 else award * 0.3
        pv_state = bm.prospect_theory_valuation(amt, reference_point=state_ref)
        inv_values.append(pv_inv["prospect_value"])
        state_values.append(pv_state["prospect_value"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=amounts / 1e6,
        y=inv_values,
        mode="lines",
        name="Investor Utility",
        line=dict(color=PRIMARY, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=amounts / 1e6,
        y=state_values,
        mode="lines",
        name="State Utility",
        line=dict(color=ACCENT, width=2.5, dash="dash"),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=TEXT_DARK, line_width=1)
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title="Prospect Theory Utility Curves",
        xaxis_title="Settlement Amount ($M)",
        yaxis_title="Subjective Prospect Value",
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def make_decision_treemap(tree_node: dict) -> go.Figure:
    """Flatten decision tree into a Plotly treemap."""
    ids, labels, parents, values, colors = [], [], [], [], []
    colour_map = {
        "Full/Partial Recovery": "#4CAF7D",
        "Discounted Recovery": "#E0B839",
        "Delayed Recovery (discounted by ~24-month delay)": "#E07B39",
        "Zero Recovery; potential cost award against investor": ACCENT,
        "Zero Recovery (award annulled); possible resubmission": ACCENT,
        "Zero / Negligible Recovery": "#CCCCCC",
    }

    def traverse(node, parent_id=""):
        nid = node["node_id"]
        label = node["label"]
        prob = node.get("probability", 1.0)
        outcome = node.get("outcome", "")
        ids.append(nid)
        labels.append(f"{label}<br>p={prob:.1%}")
        parents.append(parent_id)
        values.append(max(int(prob * 1000), 10))
        colors.append(colour_map.get(outcome, PRIMARY if "INVESTOR" in nid.upper() else CARD_BG))
        for child in node.get("children", []):
            traverse(child, nid)

    traverse(tree_node)

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors, line=dict(width=1, color=SECONDARY)),
        textinfo="label",
        hovertemplate="<b>%{label}</b><extra></extra>",
        maxdepth=4,
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title="Enforcement Decision Tree",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def make_country_comparison_radar(selected: str) -> go.Figure:
    """Compare selected country vs regional average vs best/worst."""
    all_countries = list(COUNTRY_PROFILES.keys())

    def get_scores(c: str) -> list[float]:
        p = COUNTRY_PROFILES[c]
        wjp = (p.get("wjp_score") or 0.5) * 100
        comp_map = {"Yes": 90.0, "Partial": 55.0, "No": 10.0, "Unknown": 45.0}
        return [
            p.get("wgi_rule_of_law", 50),
            p.get("wgi_corruption", 50),
            p.get("wgi_govt_effectiveness", 50),
            wjp,
            comp_map.get(p.get("voluntary_compliance_history", "Unknown"), 45),
        ]

    all_scores = [get_scores(c) for c in all_countries]
    avg_scores = [sum(s[i] for s in all_scores) / len(all_scores) for i in range(5)]
    best_scores = [max(s[i] for s in all_scores) for i in range(5)]
    worst_scores = [min(s[i] for s in all_scores) for i in range(5)]

    cats = ["Rule of Law", "Corruption Control", "Govt Effectiveness", "WJP Score", "Compliance"]
    cats_c = cats + [cats[0]]

    def close(vals: list) -> list:
        return vals + [vals[0]]

    sel_sc = get_scores(selected)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=close(worst_scores), theta=cats_c, fill="toself",
        fillcolor="rgba(200,200,200,0.15)", line=dict(color="#CCCCCC", dash="dot"),
        name="Worst in Region",
    ))
    fig.add_trace(go.Scatterpolar(
        r=close(avg_scores), theta=cats_c, fill="toself",
        fillcolor="rgba(224,184,57,0.15)", line=dict(color="#E0B839", dash="dash"),
        name="Regional Average",
    ))
    fig.add_trace(go.Scatterpolar(
        r=close(best_scores), theta=cats_c, fill="toself",
        fillcolor="rgba(76,175,125,0.15)", line=dict(color="#4CAF7D", dash="dot"),
        name="Best in Region",
    ))
    fig.add_trace(go.Scatterpolar(
        r=close(sel_sc), theta=cats_c, fill="toself",
        fillcolor=f"rgba(32,128,141,0.35)", line=dict(color=PRIMARY, width=2.5),
        name=selected,
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor=CARD_BG),
        title=f"Country Comparison: {selected} vs Regional",
        height=450,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def make_overclaiming_chart(oc_result: dict) -> go.Figure:
    claimed = oc_result["claimed_amount"]
    rational = oc_result["rational_expectation"]
    premium = oc_result["anchoring_premium"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Rational Expectation", "Overclaim Premium", "Total Claimed"],
        y=[rational / 1e6, premium / 1e6, claimed / 1e6],
        marker_color=[PRIMARY, ACCENT, SECONDARY],
        text=[fmt_usd(rational), fmt_usd(premium), fmt_usd(claimed)],
        textposition="auto",
        hovertemplate="%{x}: $%{y:.1f}M<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=f"Overclaiming Analysis — {oc_result['overclaiming_level']} Level",
        yaxis_title="USD ($M)",
        height=360,
        margin=dict(l=10, r=10, t=50, b=40),
        showlegend=False,
    )
    return fig


def make_npv_decay_chart(di_result: dict) -> go.Figure:
    years = [0, 1, 2, 3, 5, 7, 10]
    award = di_result["award_amount"]
    enf = di_result["enforcement_probability"]
    disc = 0.08
    npvs = [award * enf / ((1 + disc) ** t) for t in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=[v / 1e6 for v in npvs],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=8, color=ACCENT),
        name="NPV of Award",
        hovertemplate="Year %{x}: $%{y:.1f}M<extra></extra>",
    ))
    fig.add_hline(
        y=award * enf / 1e6,
        line_dash="dash", line_color=PRIMARY,
        annotation_text=f"Year-0 Value: {fmt_usd(award * enf)}",
        annotation_position="top right",
        annotation_font_color=PRIMARY,
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        title="NPV Decay — State's Delay Incentive",
        xaxis_title="Years of Delay",
        yaxis_title="Effective Liability ($M)",
        height=360,
        margin=dict(l=10, r=10, t=50, b=40),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size:2rem;">⚖️</div>
        <div style="font-size:1.1rem; font-weight:800; color:#A8D8DC; letter-spacing:0.05em;">
            Investor-State Dispute<br>Settlement Analysis Engine
        </div>
        <div style="font-size:0.7rem; color:rgba(255,255,255,0.55); margin-top:4px;">
            Professional dashboard for investor-state dispute settlement analysis, 
            Monte Carlo simulation, and enforcement pathway mapping for African states.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    with st.expander("📚 Data Sources", expanded=False):
        st.markdown("""
        - [ICSID Caseload Stats 2025](https://icsid.worldbank.org/sites/default/files/publications/2025-1%20ENG%20-%20The%20ICSID%20Caseload%20Statistics%20(Issue%202025-1).pdf)
        - [UNCTAD Investment Policy Hub](https://investmentpolicy.unctad.org/)
        - [italaw.com](https://www.italaw.com/)
        - [BIICL Annulment Study 2021](https://www.biicl.org/documents/10899_annulment-in-icsid-arbitration190821.pdf)
        - [Public Citizen ISDS (Dec 2024)](https://www.citizen.org/article/the-scramble-for-africa-continues-impacts-of-investor-state-dispute-settlement-on-african-countries/)
        - [World Bank WGI 2023](https://databank.worldbank.org/embed/WGI-Table/id/ceea4d8b)
        - [WJP Rule of Law Index 2023](https://worldjusticeproject.org/rule-of-law-index/downloads/WJPIndex2023.pdf)
        """)

    with st.expander("📖 Methodology", expanded=False):
        st.markdown("""
        **Monte Carlo Simulation (n=1,000–50,000 draws)**

        The engine models five multiplicative components of recovery:

        1. **Jurisdictional success** — Bernoulli draws around a calibrated base rate (global ICSID 57%), adjusted for country friction level, sector, treaty basis, and historical rates.

        2. **Award-to-claim ratio** — Sampled from the ICSID historical bracket distribution (2025) with sector-specific scale factors.

        3. **Annulment risk** — Application probability (44.7% base), annulment success rate (5% global), and expected delay (log-normal, μ=24m).

        4. **Enforcement probability** — Mapped from sovereign friction score and country profile.

        5. **Settlement discount** — Beta-distributed discount drawn from country-specific range.

        **Behavioral Models**
        - *Overclaiming bias*: Anchoring analysis comparing claim to sector rational expectation.
        - *State delay incentive*: NPV decay model with political risk premium.
        - *ZOPA analysis*: Investor floor and state ceiling based on discounted expected values.
        - *Prospect theory*: Kahneman & Tversky (1992) value function (α=0.88, λ=2.25).

        **Sources**: ICSID, UNCTAD, BIICL, Public Citizen, World Bank WGI, WJP.
        """)

    st.divider()

    # Export section
    st.markdown("**📥 Export**")
    cases_df = load_cases_df()
    st.download_button(
        label="Download Case Database (CSV)",
        data=cases_df.to_csv(index=False).encode("utf-8"),
        file_name="isds_africa_case_database.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.markdown(
        '<div style="font-size:0.7rem; color:rgba(255,255,255,0.4); text-align:center;">'
        'Updated in 2026<br>'
        'For research purposes.<br>'
    
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# MAIN CONTENT TABS
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Case Database",
    "🎲 Dispute Simulator",
    "🗺️ Enforcement Pathways",
    "🧠 Decision Engine",
    "🌍 Country Risk Profiles",
    "🧬 Behavioral Analysis",
])

# ===========================================================================
# TAB 1: CASE DATABASE EXPLORER
# ===========================================================================

with tab1:
    st.markdown("## Case Database Explorer")
    st.caption("Interactive database of ISDS cases involving African states. Sources: ICSID, UNCTAD, italaw.")

    cases_df = load_cases_df()
    stats = get_summary_stats(cases_df)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Cases", stats["total_cases"])
    with c2:
        st.metric("Investor Win Rate", fmt_pct(stats["investor_win_rate"]),
                  help="Share of resolved (non-pending) cases won by investor")
    with c3:
        st.metric("Avg Award/Claim Ratio", fmt_pct(stats["avg_award_to_claim"]),
                  help="Mean ratio where both claim and award amounts are known")
    with c4:
        st.metric("Total Amount Claimed", fmt_usd(stats["total_claimed"] * 1e6),
                  help="Sum of disclosed claim amounts")

    st.divider()

    # Filters
    section_header("Filters")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        all_countries = sorted(cases_df["Country"].unique().tolist())
        sel_countries = st.multiselect("Respondent State", all_countries, placeholder="All states")
    with fc2:
        all_sectors = sorted(cases_df["Sector"].unique().tolist())
        sel_sectors = st.multiselect("Sector", all_sectors, placeholder="All sectors")
    with fc3:
        all_outcomes = sorted(cases_df["Outcome"].unique().tolist())
        sel_outcomes = st.multiselect("Outcome", all_outcomes, placeholder="All outcomes")
    with fc4:
        yr_min = int(cases_df["Year Filed"].dropna().min())
        yr_max = int(cases_df["Year Filed"].dropna().max())
        yr_range = st.slider("Year Filed", yr_min, yr_max, (yr_min, yr_max))

    # Apply filters
    filt = cases_df.copy()
    if sel_countries:
        filt = filt[filt["Country"].isin(sel_countries)]
    if sel_sectors:
        filt = filt[filt["Sector"].isin(sel_sectors)]
    if sel_outcomes:
        filt = filt[filt["Outcome"].isin(sel_outcomes)]
    filt = filt[
        filt["Year Filed"].between(yr_range[0], yr_range[1])
        | filt["Year Filed"].isna()
    ]

    st.caption(f"Showing {len(filt)} of {len(cases_df)} cases")

    display_cols = [
        "Case Name", "Country", "Sector", "Treaty Basis", "Year Filed",
        "Outcome", "Amount Claimed ($M)", "Amount Awarded ($M)", "Annulment Attempted",
    ]
    st.dataframe(
        filt[display_cols],
        use_container_width=True,
        height=380,
        column_config={
            "Amount Claimed ($M)": st.column_config.NumberColumn(format="$%.1f M"),
            "Amount Awarded ($M)": st.column_config.NumberColumn(format="$%.1f M"),
            "Year Filed": st.column_config.NumberColumn(format="%d"),
        },
    )

    st.divider()

    # Charts row
    section_header("Visualisations")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(make_bar_cases_by_country(filt), use_container_width=True)
    with ch2:
        st.plotly_chart(make_donut_outcomes(filt), use_container_width=True)

    st.plotly_chart(make_timeline_filings(filt), use_container_width=True)


# ===========================================================================
# TAB 2: DISPUTE SIMULATOR
# ===========================================================================

with tab2:
    st.markdown("## Dispute Simulator")
    st.caption("Monte Carlo simulation of ISDS dispute outcomes. Configure parameters and run simulation.")

    # ── Input Panel ──────────────────────────────────────────────────────────
    section_header("Dispute Parameters")

    inp1, inp2, inp3 = st.columns([2, 2, 3])

    with inp1:
        sel_state = st.selectbox(
            "Respondent State",
            sorted(COUNTRY_PROFILES.keys()),
            index=list(sorted(COUNTRY_PROFILES.keys())).index("Tanzania"),
        )
        sel_sector = st.selectbox(
            "Sector",
            list(SECTOR_STATS.keys()),
            index=list(SECTOR_STATS.keys()).index("Mining"),
        )
        sel_treaty = st.selectbox(
            "Treaty Basis",
            list(TREATY_BASIS_STATS.keys()),
        )

    with inp2:
        sel_investor = st.selectbox(
            "Investor Nationality",
            ["UK", "USA", "Germany", "France", "Netherlands", "China",
             "Canada", "Australia", "UAE", "South Africa", "Other"],
        )
        sel_claimed = st.number_input(
            "Amount Claimed (USD)",
            min_value=1_000_000,
            max_value=30_000_000_000,
            value=500_000_000,
            step=10_000_000,
            format="%d",
            help="Enter the amount claimed by the investor in USD",
        )
        investment_type = st.text_input(
            "Investment Type (optional)",
            value="greenfield mining",
            help="e.g. 'greenfield mining', 'share acquisition', 'infrastructure concession'",
        )

    with inp3:
        n_sims = st.slider(
            "Number of Simulations",
            min_value=1_000,
            max_value=50_000,
            value=10_000,
            step=1_000,
            help="More simulations → more accurate distributions, but slower",
        )
        use_seed = st.checkbox("Use fixed random seed (reproducible)", value=True)
        seed_val = st.number_input("Seed", value=42, min_value=0, max_value=99999) if use_seed else None

        st.markdown("")
        run_btn = st.button("▶ Run Simulation", use_container_width=True, type="primary")

    # ── Run Simulation ────────────────────────────────────────────────────────
    if run_btn:
        dp = DisputeProfile(
            respondent_state=sel_state,
            investor_nationality=sel_investor,
            sector=sel_sector,
            treaty_basis=sel_treaty,
            amount_claimed_usd=float(sel_claimed),
            investment_type=investment_type,
        )

        with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
            engine = SimulationEngine(dp, n_simulations=n_sims, seed=seed_val)
            full_results = engine.run_full_simulation()

            # Also get raw distributions (not stripped by run_full_simulation)
            recovery_sim = engine.simulate_recovery_rate()
            award_sim = engine.simulate_award_to_claim_ratio()
            timeline_sim = engine.simulate_enforcement_timeline()

        # Store in session state
        st.session_state["sim_results"] = full_results
        st.session_state["sim_dp"] = dp
        st.session_state["sim_recovery_dist"] = recovery_sim["distribution"]
        st.session_state["sim_award_dist"] = award_sim["distribution"]
        st.session_state["sim_timeline_dist"] = timeline_sim["timeline_months_distribution"]
        st.session_state["sim_country_profile"] = COUNTRY_PROFILES.get(sel_state, {})

        st.success("Simulation complete!")

    # ── Results ────────────────────────────────────────────────────────────────
    if "sim_results" in st.session_state:
        res = st.session_state["sim_results"]
        dp_stored = st.session_state["sim_dp"]
        summary = res["summary"]
        jurisd = res["jurisdictional_success"]
        ann = res["annulment_risk"]
        recovery_dist = st.session_state["sim_recovery_dist"]
        award_dist = st.session_state["sim_award_dist"]
        timeline_dist = st.session_state["sim_timeline_dist"]
        cp = st.session_state["sim_country_profile"]

        st.divider()
        section_header("Simulation Results")

        # ── Row 1: KPI Cards ──────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric(
                "Jurisdictional Success",
                fmt_pct(summary["jurisdictional_success_prob"]),
                delta=f"CI: {fmt_pct(jurisd['confidence_interval'][0])}–{fmt_pct(jurisd['confidence_interval'][1])}",
            )
        with k2:
            exp_award = summary["expected_recovery_usd"]
            p25_award = dp_stored.amount_claimed_usd * res["recovery_rate"]["percentiles"]["p25"]
            p75_award = dp_stored.amount_claimed_usd * res["recovery_rate"]["percentiles"]["p75"]
            st.metric(
                "Expected Recovery (USD)",
                fmt_usd(exp_award),
                delta=f"P25–P75: {fmt_usd(p25_award)}–{fmt_usd(p75_award)}",
            )
        with k3:
            st.metric(
                "Annulment Risk",
                fmt_pct(summary["annulment_net_risk"]),
                delta=f"App. rate: {fmt_pct(ann['application_probability'])}",
            )
        with k4:
            st.metric(
                "Sovereign Friction Score",
                f"{summary['sovereign_friction_score']:.0f}/100",
                delta=cp.get("enforcement_friction_level", ""),
            )
        with k5:
            rec_mean = summary["expected_recovery_fraction"]
            rec_p25 = res["recovery_rate"]["percentiles"]["p25"]
            rec_p75 = res["recovery_rate"]["percentiles"]["p75"]
            st.metric(
                "Expected Recovery Rate",
                fmt_pct(rec_mean),
                delta=f"P25–P75: {fmt_pct(rec_p25)}–{fmt_pct(rec_p75)}",
            )

        st.divider()

        # ── Row 2: Histogram charts ───────────────────────────────────────────
        ch_a, ch_b = st.columns(2)
        with ch_a:
            st.plotly_chart(make_recovery_histogram(recovery_dist), use_container_width=True)
        with ch_b:
            st.plotly_chart(
                make_award_histogram(award_dist, dp_stored.amount_claimed_usd),
                use_container_width=True,
            )

        # ── Row 3: Timeline + Radar ───────────────────────────────────────────
        ch_c, ch_d = st.columns(2)
        with ch_c:
            st.plotly_chart(make_timeline_box(timeline_dist), use_container_width=True)
        with ch_d:
            st.plotly_chart(make_radar_friction(dp_stored.respondent_state), use_container_width=True)

        # ── Row 4: Settlement Zone ────────────────────────────────────────────
        section_header("Settlement Zone (ZOPA) Analysis")
        bm = BehavioralModule()
        enf_prob = summary["enforcement_prob"]
        j_prob = summary["jurisdictional_success_prob"]
        median_award = dp_stored.amount_claimed_usd * res["award_to_claim"]["median"]

        zopa = bm.calculate_settlement_zone(
            award_amount=median_award,
            investor_recovery_prob=j_prob * enf_prob,
            state_enforcement_risk=enf_prob,
        )
        st.plotly_chart(
            make_settlement_bar(zopa["investor_floor"], zopa["state_ceiling"], median_award),
            use_container_width=True,
        )
        zopa_css = "risk-low" if zopa["settlement_zone_exists"] else "risk-high"
        info_card(zopa["commentary"], zopa_css)

        # ── Row 5: Prospect Theory ────────────────────────────────────────────
        section_header("Prospect Theory Valuation")
        st.caption("How each party subjectively values the same settlement amount under Kahneman & Tversky's loss-aversion model (α=0.88, λ=2.25).")
        st.plotly_chart(
            make_prospect_theory_chart(median_award, zopa["investor_floor"], zopa["state_ceiling"]),
            use_container_width=True,
        )

        # ── Simulation Details Expander ───────────────────────────────────────
        with st.expander("📋 Full Simulation Details"):
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Component Means (Recovery)**")
                comp = res["recovery_rate"].get("component_means", {})
                for k, v in comp.items():
                    st.write(f"- **{k.replace('_', ' ').title()}**: {fmt_pct(v)}")
            with rc2:
                st.markdown("**Award Bracket Distribution (ICSID historical)**")
                for label, share in res["award_to_claim"]["bracket_probs"].items():
                    st.write(f"- {label}: **{share:.0f}%**")

    else:
        st.info("Configure dispute parameters above and click **▶ Run Simulation** to see results.")


# ===========================================================================
# TAB 3: ENFORCEMENT PATHWAY MAPPER
# ===========================================================================

with tab3:
    st.markdown("## Enforcement Pathway Mapper")
    st.caption("Multi-jurisdiction enforcement strategy and asset attachability analysis.")

    # Use simulation results if available, otherwise allow standalone input
    has_sim = "sim_results" in st.session_state

    if has_sim:
        st.success("Using parameters from the last Dispute Simulator run. You can also customise below.")

    ep_c1, ep_c2, ep_c3 = st.columns(3)
    with ep_c1:
        ep_state = st.selectbox(
            "Respondent State",
            sorted(COUNTRY_PROFILES.keys()),
            index=(
                list(sorted(COUNTRY_PROFILES.keys())).index(
                    st.session_state.get("sim_dp", None) and
                    st.session_state["sim_dp"].respondent_state or "Tanzania"
                )
                if has_sim else
                list(sorted(COUNTRY_PROFILES.keys())).index("Tanzania")
            ),
            key="ep_state",
        )
    with ep_c2:
        ep_award = st.number_input(
            "Award Amount (USD)",
            min_value=100_000,
            max_value=10_000_000_000,
            value=(
                int(st.session_state["sim_results"]["summary"]["expected_recovery_usd"])
                if has_sim else 100_000_000
            ),
            step=1_000_000,
            format="%d",
            key="ep_award",
        )
    with ep_c3:
        st.markdown("")
        st.markdown("")
        map_btn = st.button("🗺️ Map Enforcement Pathways", use_container_width=True, key="map_btn")

    if map_btn or has_sim:
        ep_profile = COUNTRY_PROFILES.get(ep_state, {})
        ep_obj = EnforcementPathway(ep_profile, float(ep_award), ep_state)

        # ── Jurisdiction Table ─────────────────────────────────────────────
        section_header("Ranked Enforcement Jurisdictions")
        jurisdictions = ep_obj.map_jurisdictions()

        jur_rows = []
        for j in jurisdictions:
            jur_rows.append({
                "Rank": jurisdictions.index(j) + 1,
                "Jurisdiction": j["jurisdiction"],
                "Success Probability": j["success_probability"],
                "Timeline (months)": j["timeline_months"],
                "Est. Cost": fmt_usd(j["costs_estimate_usd"]),
                "Rationale": j["rationale"][:120] + "...",
            })
        jur_df = pd.DataFrame(jur_rows)
        st.dataframe(
            jur_df[["Rank", "Jurisdiction", "Success Probability", "Timeline (months)", "Est. Cost"]],
            use_container_width=True,
            column_config={
                "Success Probability": st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="%.0%",
                ),
            },
        )

        # Bar chart of jurisdiction probabilities
        fig_jur = px.bar(
            jur_df,
            x="Success Probability",
            y="Jurisdiction",
            orientation="h",
            color="Success Probability",
            color_continuous_scale=[[0, CARD_BG], [1, PRIMARY]],
            title="Enforcement Success Probability by Jurisdiction",
            text="Success Probability",
        )
        fig_jur.update_traces(texttemplate="%{text:.0%}", textposition="outside")
        fig_jur.update_layout(**PLOTLY_TEMPLATE, coloraxis_showscale=False, height=320,
                               margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_jur, use_container_width=True)

        # ── Asset Attachability ───────────────────────────────────────────
        st.divider()
        section_header("Asset Attachability Assessment")

        att = ep_obj.score_asset_attachability()
        att_c1, att_c2 = st.columns([1, 2])

        with att_c1:
            # Gauge chart
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=att["score"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Attachability Score<br>Grade: {att['grade']}",
                       "font": {"size": 14, "color": TEXT_DARK}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PRIMARY},
                    "steps": [
                        {"range": [0, 30], "color": "#FFE5E0"},
                        {"range": [30, 60], "color": "#FFF9E0"},
                        {"range": [60, 100], "color": "#E5F5EE"},
                    ],
                    "threshold": {
                        "line": {"color": ACCENT, "width": 3},
                        "thickness": 0.75,
                        "value": att["score"],
                    },
                },
            ))
            gauge_fig.update_layout(**PLOTLY_TEMPLATE, height=280,
                                     margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Component scores
            comp = att["component_scores"]
            comp_df = pd.DataFrame({
                "Component": ["SWF Assets (max 35)", "SOE Assets (max 30)",
                               "ICSID Membership (max 15)", "Compliance History (max 20)"],
                "Score": [comp["swf_score"], comp["soe_score"],
                          comp["icsid_score"], comp["compliance_score"]],
            })
            fig_comp = px.bar(
                comp_df, x="Score", y="Component", orientation="h",
                color="Score", color_continuous_scale=[[0, CARD_BG], [1, PRIMARY]],
            )
            fig_comp.update_layout(**PLOTLY_TEMPLATE, coloraxis_showscale=False,
                                    height=200, margin=dict(l=10, r=10, t=10, b=10),
                                    showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

        with att_c2:
            st.markdown("**Key Target Assets**")
            for t in att["key_targets"]:
                info_card(f"🎯 {t}", "low")

            st.markdown("")
            st.markdown("**Immunity Risks**")
            for r in att["immunity_risks"]:
                info_card(f"⚠️ {r}", "high")

        # ── Sequencing ────────────────────────────────────────────────────
        st.divider()
        section_header("Recommended Enforcement Sequencing")
        seq = ep_obj.recommend_sequencing()
        priority_colours = {
            "Critical": "#A84B2F", "High": "#E07B39",
            "Medium": "#E0B839", "Low": "#4CAF7D"
        }
        for step in seq:
            pcolor = priority_colours.get(step.get("priority", "Medium"), PRIMARY)
            with st.expander(
                f"Step {step['step']}: {step['phase']} — {step.get('priority', '')} Priority",
                expanded=(step["step"] <= 2),
            ):
                st.markdown(f"**Action:** {step['action']}")
                st.markdown(f"**Timeline:** {step['timeline']}")
                st.markdown(f"**Rationale:** {step['rationale']}")
                st.markdown(f"<span style='color:{pcolor}; font-weight:700;'>Priority: {step.get('priority', '')}</span>",
                            unsafe_allow_html=True)
    else:
        st.info("Run a simulation first or configure the parameters above and click **Map Enforcement Pathways**.")


# ===========================================================================
# TAB 4: DECISION ENGINE
# ===========================================================================

with tab4:
    st.markdown("## Decision Engine")
    st.caption("Decision tree, settlement threshold calculator, and advisory brief generation.")

    has_sim = "sim_results" in st.session_state

    de_c1, de_c2 = st.columns(2)
    with de_c1:
        de_state = st.selectbox(
            "Respondent State",
            sorted(COUNTRY_PROFILES.keys()),
            index=(
                list(sorted(COUNTRY_PROFILES.keys())).index(
                    st.session_state["sim_dp"].respondent_state
                ) if has_sim else
                list(sorted(COUNTRY_PROFILES.keys())).index("Tanzania")
            ),
            key="de_state",
        )
    with de_c2:
        de_award = st.number_input(
            "Award Amount (USD)",
            min_value=100_000,
            max_value=10_000_000_000,
            value=(
                int(st.session_state["sim_results"]["summary"]["expected_recovery_usd"])
                if has_sim else 100_000_000
            ),
            step=1_000_000,
            format="%d",
            key="de_award",
        )

    gen_btn = st.button("🌳 Generate Decision Tree & Brief", use_container_width=False, key="gen_tree")

    if gen_btn or has_sim:
        de_profile = COUNTRY_PROFILES.get(de_state, {})
        de_ep = EnforcementPathway(de_profile, float(de_award), de_state)

        # ── Decision Tree ──────────────────────────────────────────────────
        section_header("Decision Tree")
        tree = de_ep.generate_decision_tree()
        st.plotly_chart(make_decision_treemap(tree), use_container_width=True)

        # ── Settlement Threshold Calculator ───────────────────────────────
        section_header("Settlement Threshold Calculator")
        bm = BehavioralModule()

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            inv_disc = st.slider("Investor Discount Rate", 0.05, 0.25, 0.10, 0.01,
                                 format="%.0f%%", key="inv_disc")
        with sc2:
            state_disc = st.slider("State Discount Rate", 0.03, 0.15, 0.06, 0.01,
                                   format="%.0f%%", key="state_disc")
        with sc3:
            enf_prob_val = de_profile.get("enforcement_friction_level", "Moderate")
            enf_num = {
                "Critical": 0.15, "Very High": 0.25, "High": 0.45,
                "Moderate": 0.65, "Moderate-Low": 0.80, "Low": 0.92
            }.get(enf_prob_val, 0.65)
            j_prob_val = 0.57  # default if no simulation
            if has_sim:
                j_prob_val = st.session_state["sim_results"]["summary"]["jurisdictional_success_prob"]
            st.metric("Enforcement Friction", enf_prob_val)
            st.metric("Enforcement Probability", f"{enf_num:.0%}")

        zopa = bm.calculate_settlement_zone(
            award_amount=float(de_award),
            investor_recovery_prob=j_prob_val * enf_num,
            state_enforcement_risk=enf_num,
            investor_discount_rate=inv_disc,
            state_discount_rate=state_disc,
        )

        zt1, zt2, zt3 = st.columns(3)
        with zt1:
            st.metric("Investor Floor", fmt_usd(zopa["investor_floor"]))
        with zt2:
            st.metric("State Ceiling", fmt_usd(zopa["state_ceiling"]))
        with zt3:
            zopa_mid = zopa.get("zopa_midpoint")
            st.metric("ZOPA Midpoint", fmt_usd(zopa_mid) if zopa_mid else "No ZOPA")

        info_card(zopa["commentary"],
                  "low" if zopa["settlement_zone_exists"] else "high")

        # ── Generate Brief ─────────────────────────────────────────────────
        section_header("Advisory Brief Generator")

        if has_sim:
            brief_btn = st.button("📄 Generate Advisory Brief", key="brief_btn")
            if brief_btn:
                with st.spinner("Generating advisory memo..."):
                    dp_stored = st.session_state["sim_dp"]
                    cp_stored = st.session_state["sim_country_profile"]
                    res_stored = st.session_state["sim_results"]
                    ep_memo = EnforcementPathway(cp_stored, float(de_award), de_state)
                    mg = MemoGenerator(res_stored, dp_stored, cp_stored, ep_memo)
                    full_memo = mg.generate_full_memo()
                    csv_export = mg.generate_csv_export()

                st.session_state["memo_text"] = full_memo
                st.session_state["memo_csv"] = csv_export

        if "memo_text" in st.session_state:
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="📥 Download Advisory Memo (.txt)",
                    data=st.session_state["memo_text"].encode("utf-8"),
                    file_name=f"ISDS_Advisory_Memo_{de_state.replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    label="📥 Download Simulation Results (.csv)",
                    data=st.session_state["memo_csv"].encode("utf-8"),
                    file_name=f"ISDS_Simulation_{de_state.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with st.expander("📄 Preview Advisory Memo", expanded=False):
                st.code(st.session_state["memo_text"][:8000] + "\n\n[...truncated for preview...]",
                        language=None)
        else:
            if not has_sim:
                st.info("Run a simulation in the **Dispute Simulator** tab first to enable brief generation.")
            else:
                st.info("Click **Generate Advisory Brief** above to produce the downloadable memo.")

    else:
        st.info("Configure parameters above and click **Generate Decision Tree & Brief**.")


# ===========================================================================
# TAB 5: COUNTRY RISK PROFILES
# ===========================================================================

with tab5:
    st.markdown("## Country Risk Profiles")
    st.caption("Detailed governance, enforcement, and case history analysis for each African state.")

    cr_state = st.selectbox(
        "Select Country",
        sorted(COUNTRY_PROFILES.keys()),
        index=list(sorted(COUNTRY_PROFILES.keys())).index("Tanzania"),
        key="cr_state",
    )

    cp = COUNTRY_PROFILES[cr_state]
    cases_df = load_cases_df()
    country_cases = cases_df[cases_df["Country"] == cr_state]

    # Profile header row
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    with pc1:
        st.metric("Rule of Law (WGI %ile)", f"{cp.get('wgi_rule_of_law', 'N/A')}")
    with pc2:
        st.metric("Corruption Control (%ile)", f"{cp.get('wgi_corruption', 'N/A')}")
    with pc3:
        st.metric("Govt. Effectiveness (%ile)", f"{cp.get('wgi_govt_effectiveness', 'N/A')}")
    with pc4:
        wjp = cp.get("wjp_score")
        st.metric("WJP Rule of Law", f"{wjp:.2f}" if wjp else "N/A")
    with pc5:
        st.metric("Enforcement Friction", cp.get("enforcement_friction_level", "N/A"))

    # Profile details
    section_header(f"Profile: {cr_state}")
    pd1, pd2 = st.columns(2)
    with pd1:
        friction = cp.get("enforcement_friction_level", "N/A")
        friction_css = {
            "Critical": "risk-critical", "Very High": "risk-high",
            "High": "risk-high", "Moderate": "risk-moderate",
            "Moderate-Low": "risk-low", "Low": "risk-low"
        }.get(friction, "")

        info_card(f"<b>ICSID Member:</b> {cp.get('icsid_member', 'N/A')}", friction_css)
        info_card(f"<b>Voluntary Compliance:</b> {cp.get('voluntary_compliance_history', 'Unknown')}", friction_css)
        disc = cp.get("settlement_discount_range", (0.10, 0.30))
        info_card(f"<b>Settlement Discount Range:</b> {disc[0]:.0%} – {disc[1]:.0%}")
        info_card(f"<b>Avg. Settlement Timeline:</b> {cp.get('avg_settlement_years', 'N/A')} years")

        swf_name = cp.get("swf_name")
        swf_aum = cp.get("swf_aum_billions")
        if swf_name:
            info_card(f"<b>Sovereign Wealth Fund:</b> {swf_name} (~${swf_aum:.1f}B AUM)", "low")
        else:
            info_card("<b>Sovereign Wealth Fund:</b> None identified")

    with pd2:
        soes = cp.get("major_soes", [])
        st.markdown("**Major State-Owned Enterprises (SOEs)**")
        for soe in soes:
            info_card(f"🏛️ {soe}")

    # Historical cases
    st.divider()
    section_header(f"Historical Cases — {cr_state}")
    if len(country_cases) > 0:
        hist_stats = calculate_historical_rates(cr_state)

        hc1, hc2, hc3, hc4 = st.columns(4)
        with hc1:
            st.metric("Total Cases", hist_stats["total_cases"])
        with hc2:
            st.metric("Investor Win Rate", fmt_pct(hist_stats["investor_win_rate"]))
        with hc3:
            st.metric("Settlement Rate", fmt_pct(hist_stats["settlement_rate"]))
        with hc4:
            st.metric("Avg Award/Claim", fmt_pct(hist_stats["avg_award_to_claim_ratio"]))

        st.dataframe(
            country_cases[[
                "Case Name", "Sector", "Treaty Basis", "Year Filed", "Outcome",
                "Amount Claimed ($M)", "Amount Awarded ($M)", "Enforcement Status"
            ]],
            use_container_width=True,
            height=300,
        )
    else:
        st.info(f"No cases recorded for {cr_state} in the database.")

    # Comparison chart
    st.divider()
    section_header("Regional Comparison")
    st.plotly_chart(make_country_comparison_radar(cr_state), use_container_width=True)

    # Enforcement friction gauge
    section_header("Enforcement Friction Score")
    dp_temp = DisputeProfile(
        respondent_state=cr_state,
        investor_nationality="UK",
        sector="Mining",
        treaty_basis="Bilateral Investment Treaty",
        amount_claimed_usd=100_000_000,
    )
    eng_temp = SimulationEngine(dp_temp, n_simulations=100)
    friction_score = eng_temp.score_sovereign_friction()

    gauge_fr = go.Figure(go.Indicator(
        mode="gauge+number",
        value=friction_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Sovereign Friction Score — {cr_state}", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": ACCENT if friction_score > 60 else PRIMARY},
            "steps": [
                {"range": [0, 35], "color": "#E5F5EE"},
                {"range": [35, 65], "color": "#FFF9E0"},
                {"range": [65, 100], "color": "#FFE5E0"},
            ],
        },
    ))
    gauge_fr.update_layout(**PLOTLY_TEMPLATE, height=300, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(gauge_fr, use_container_width=True)


# ===========================================================================
# TAB 6: BEHAVIORAL ANALYSIS
# ===========================================================================

with tab6:
    st.markdown("## Behavioral Analysis")
    st.caption("Overclaiming bias, prospect theory, and state delay incentive modelling.")

    bm = BehavioralModule()

    section_header("1 · Overclaiming Bias Analysis")

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        oc_sector = st.selectbox("Sector", list(SECTOR_STATS.keys()),
                                 index=list(SECTOR_STATS.keys()).index("Mining"), key="oc_sector")
    with oc2:
        oc_claimed = st.number_input(
            "Claimed Amount (USD)", min_value=1_000_000, max_value=30_000_000_000,
            value=500_000_000, step=10_000_000, format="%d", key="oc_claimed"
        )
    with oc3:
        oc_custom_ratio = st.number_input(
            "Custom Avg Award Ratio (optional, 0=use sector default)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="oc_ratio"
        )

    oc_result = bm.analyze_overclaiming_bias(
        float(oc_claimed),
        oc_sector,
        historical_avg_ratio=(oc_custom_ratio if oc_custom_ratio > 0 else None),
    )

    ock1, ock2, ock3 = st.columns(3)
    with ock1:
        st.metric("Rational Expectation", fmt_usd(oc_result["rational_expectation"]))
    with ock2:
        st.metric("Overclaiming Factor", f"{oc_result['overclaiming_factor']:.2f}×")
    with ock3:
        st.metric("Anchoring Premium", fmt_usd(oc_result["anchoring_premium"]))

    level = oc_result["overclaiming_level"]
    css = {"Low": "risk-low", "Moderate": "risk-moderate",
           "High": "risk-high", "Extreme": "risk-critical"}.get(level, "")
    info_card(f"<b>Level: {level}</b> — {oc_result['strategic_implication']}", css)
    st.plotly_chart(make_overclaiming_chart(oc_result), use_container_width=True)

    st.divider()
    section_header("2 · State Delay Incentive (NPV Decay)")

    di1, di2, di3 = st.columns(3)
    with di1:
        di_award = st.number_input(
            "Award Amount (USD)", min_value=100_000, max_value=10_000_000_000,
            value=100_000_000, step=1_000_000, format="%d", key="di_award"
        )
    with di2:
        di_friction = st.selectbox(
            "Enforcement Friction Level",
            ["Critical", "Very High", "High", "Moderate", "Moderate-Low", "Low"],
            index=3, key="di_friction"
        )
    with di3:
        di_rate = st.slider("State Discount Rate", 0.03, 0.20, 0.08, 0.01,
                            format="%.0f%%", key="di_rate")

    di_result = bm.analyze_state_delay_incentive(float(di_award), di_friction, di_rate)

    dik1, dik2, dik3 = st.columns(3)
    with dik1:
        st.metric("5-Year Delay Incentive", fmt_usd(di_result["delay_incentive_usd"]))
    with dik2:
        st.metric("P(Never Pay)", fmt_pct(di_result["probability_of_non_payment"]))
    with dik3:
        st.metric("Optimal Delay (Years)", f"{di_result['optimal_delay_years']:.0f}")

    info_card(di_result["summary"])
    st.plotly_chart(make_npv_decay_chart(di_result), use_container_width=True)

    st.divider()
    section_header("3 · Interactive Prospect Theory Analysis")

    pt1, pt2, pt3 = st.columns(3)
    with pt1:
        pt_amount = st.number_input(
            "Settlement Amount (USD)", min_value=100_000, max_value=10_000_000_000,
            value=50_000_000, step=1_000_000, format="%d", key="pt_amount"
        )
    with pt2:
        pt_ref = st.number_input(
            "Reference Point (USD)", min_value=100_000, max_value=10_000_000_000,
            value=100_000_000, step=1_000_000, format="%d", key="pt_ref"
        )
    with pt3:
        pt_lambda = st.slider("Loss Aversion (λ)", 1.0, 5.0, 2.25, 0.05, key="pt_lambda")

    pt_result = bm.prospect_theory_valuation(float(pt_amount), float(pt_ref), pt_lambda)
    pt_css = "risk-high" if pt_result["is_loss"] else "risk-low"
    info_card(pt_result["interpretation"], pt_css)

    ptk1, ptk2, ptk3 = st.columns(3)
    with ptk1:
        st.metric("Gain / Loss vs Reference",
                  fmt_usd(pt_result["gain_or_loss"]) if pt_result["gain_or_loss"] >= 0
                  else f"-{fmt_usd(-pt_result['gain_or_loss'])}",
                  delta="GAIN" if not pt_result["is_loss"] else "LOSS")
    with ptk2:
        st.metric("Prospect Value", f"{pt_result['prospect_value']:,.1f}")
    with ptk3:
        st.metric("Loss Aversion λ", f"{pt_result['loss_aversion_lambda']:.2f}")

    # Dynamic prospect theory visualisation for current anchor
    pt_amounts_range = np.linspace(pt_ref * 0.05, pt_ref * 1.3, 300)
    pt_values_range = [
        bm.prospect_theory_valuation(a, pt_ref, pt_lambda)["prospect_value"]
        for a in pt_amounts_range
    ]
    pt_fig = go.Figure()
    pt_fig.add_trace(go.Scatter(
        x=pt_amounts_range / 1e6,
        y=pt_values_range,
        mode="lines",
        line=dict(color=PRIMARY, width=2.5),
        name="Prospect Value",
        hovertemplate="Amount: $%{x:.1f}M<br>Value: %{y:.1f}<extra></extra>",
    ))
    pt_fig.add_vline(x=pt_ref / 1e6, line_dash="dash", line_color=TEXT_DARK, line_width=1.5,
                     annotation_text=f"Reference: {fmt_usd(pt_ref)}",
                     annotation_position="top", annotation_font_color=TEXT_DARK)
    pt_fig.add_vline(x=pt_amount / 1e6, line_dash="dot", line_color=ACCENT, line_width=2,
                     annotation_text=f"Current: {fmt_usd(pt_amount)}",
                     annotation_position="top right", annotation_font_color=ACCENT)
    pt_fig.add_hline(y=0, line_dash="solid", line_color=TEXT_DARK, line_width=0.8)
    pt_fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=f"Prospect Value Curve (λ={pt_lambda:.2f}, α=0.88)",
        xaxis_title="Settlement Amount ($M)",
        yaxis_title="Subjective Prospect Value",
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(pt_fig, use_container_width=True)

    st.divider()
    section_header("4 · Anchoring Effect — How Claims Shift Tribunal Starting Points")

    anch_claim = st.slider(
        "Claimed Amount ($M)", min_value=10, max_value=5000,
        value=500, step=10, key="anch_claim",
        help="Slide to explore how different claim amounts affect the ZOPA and anchoring"
    )
    anch_sector = st.selectbox("Sector", list(SECTOR_STATS.keys()), key="anch_sector")

    anch_result = bm.analyze_overclaiming_bias(
        float(anch_claim) * 1e6, anch_sector
    )

    anchor_fig = go.Figure()
    amounts_anchor = np.linspace(0, anch_claim * 1e6, 300)
    anchor_fig.add_trace(go.Scatter(
        x=amounts_anchor / 1e6,
        y=[bm.prospect_theory_valuation(a, anch_claim * 1e6)["prospect_value"]
           for a in amounts_anchor],
        mode="lines",
        line=dict(color=PRIMARY, width=2, dash="solid"),
        name=f"Anchored to Claim ({anch_claim}M)",
        fill="tozeroy", fillcolor=f"rgba(32,128,141,0.08)",
    ))
    anchor_fig.add_trace(go.Scatter(
        x=amounts_anchor / 1e6,
        y=[bm.prospect_theory_valuation(a, anch_result["rational_expectation"])["prospect_value"]
           for a in amounts_anchor],
        mode="lines",
        line=dict(color=ACCENT, width=2, dash="dash"),
        name=f"Anchored to Rational (${anch_result['rational_expectation']/1e6:.0f}M)",
        fill="tozeroy", fillcolor=f"rgba(168,75,47,0.08)",
    ))
    anchor_fig.add_hline(y=0, line_color=TEXT_DARK, line_width=0.8)
    anchor_fig.update_layout(
        **PLOTLY_TEMPLATE,
        title=f"Anchoring Effect: Claim vs Rational Expectation ({anch_sector})",
        xaxis_title="Settlement Amount ($M)",
        yaxis_title="Subjective Value",
        height=380,
        margin=dict(l=10, r=10, t=50, b=40),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(anchor_fig, use_container_width=True)

    info_card(
        f"<b>Overclaiming Level: {anch_result['overclaiming_level']}</b> "
        f"(Factor: {anch_result['overclaiming_factor']:.2f}×) — "
        f"{anch_result['strategic_implication']}",
        {"Low": "risk-low", "Moderate": "risk-moderate",
         "High": "risk-high", "Extreme": "risk-critical"}.get(anch_result["overclaiming_level"], "")
    )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="app-footer">'
    'ISDS Recovery Realism Engine v1.0 · Data: ICSID, UNCTAD, BIICL, World Bank WGI, WJP · '
    'For research and educational purposes only. Not legal advice.'
    '</div>',
    unsafe_allow_html=True,
)
