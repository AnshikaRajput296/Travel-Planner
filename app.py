"""
app.py – AI Multi-Agent Travel Planner
=======================================
Enterprise-grade Streamlit dashboard powered by LangGraph + Groq (Llama 3.3 70B).

Run: streamlit run app.py
"""

from __future__ import annotations
import os
import json
import time
import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

from guardrails.validator import validate_inputs
from graph.travel_graph import run_travel_pipeline

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
Path("exports").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

st.set_page_config(
    page_title="VoyageAI — Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = Path("assets/styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Inline design tokens (fallback / extensions) ──────────────────────────────
st.markdown("""
<style>
/* ── Glassmorphism card helper ─────────── */
.glass-card {
  background: rgba(15,23,42,0.75);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 20px;
  padding: 1.6rem 2rem;
  margin-bottom: 1.2rem;
}
.kpi-card {
  background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(8,12,20,0.9) 100%);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 1.4rem 1.6rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.kpi-card:hover { transform: translateY(-4px); }
.kpi-card .glow-bar {
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  border-radius: 20px 20px 0 0;
}
.kpi-label {
  font-size: 0.78rem; font-weight: 600; letter-spacing: 1px;
  text-transform: uppercase; color: #94a3b8; margin-bottom: 0.5rem;
}
.kpi-value {
  font-family: 'DM Serif Display', serif; font-size: 2rem;
  font-weight: 700; line-height: 1.1;
}
.kpi-sub { font-size: 0.82rem; color: #64748b; margin-top: 0.3rem; }
/* ── Section headers ───────────────────── */
.section-title {
  font-family: 'DM Serif Display', serif;
  font-size: 1.5rem; font-weight: 700;
  color: #f1f5f9; margin: 1.5rem 0 1rem;
  display: flex; align-items: center; gap: 0.6rem;
}
/* ── Agent status pill ─────────────────── */
.agent-pill {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(6,182,212,0.12);
  border: 1px solid rgba(6,182,212,0.3);
  border-radius: 50px; padding: 6px 16px;
  font-size: 0.82rem; font-weight: 600; color: #06b6d4;
  margin: 4px; animation: fadeIn 0.4s ease;
}
.agent-pill.done { background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.3); color: #10b981; }
.agent-pill.error { background: rgba(244,63,94,0.12); border-color: rgba(244,63,94,0.3); color: #f43f5e; }
@keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
/* ── Hero banner ───────────────────────── */
.hero-banner {
  background: linear-gradient(135deg, #0d1526 0%, #0f2045 50%, #0a1628 100%);
  border: 1px solid rgba(6,182,212,0.2);
  border-radius: 24px;
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  position: relative; overflow: hidden;
}
.hero-banner::before {
  content: '';
  position: absolute; top: -60%; right: -20%;
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
  pointer-events: none;
}
/* ── Itinerary day card ─────────────────── */
.day-card {
  background: rgba(15,23,42,0.6);
  border-left: 3px solid #2B1A1A;
  border-radius: 0 12px 12px 0;
  padding: 1rem 1.5rem;
  margin-bottom: 1rem;
}
/* ── Sidebar brand ──────────────────────── */
.sidebar-brand {
  font-family: 'DM Serif Display', serif;
  font-size: 1.6rem; font-weight: 700;
  background: linear-gradient(135deg, #06b6d4, #8b5cf6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 0.2rem;
            
}
.sidebar-tagline {
  font-size: 0.78rem; color: #6748b4; letter-spacing: 1px;
  text-transform: uppercase; margin-bottom: 1.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "plan_result":    None,
    "search_history": [],
    "is_loading":     False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _build_text_report(plan: dict) -> str:
    """Build a clean text export of the travel plan."""
    bd = plan.get("budget_breakdown", {})
    lines = [
        "=" * 60,
        f"  AI TRAVEL PLAN — {plan.get('destination','').upper()}",
        "=" * 60,
        f"Duration   : {plan.get('days')} days",
        f"Travelers  : {plan.get('travelers')}",
        f"Budget     : ₹{plan.get('budget', 0):,.0f}",
        f"Generated  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "── BUDGET SUMMARY ─────────────────────────────────────",
        f"  Total Budget     : ₹{bd.get('total_budget', 0):,.0f}",
        f"  Estimated Cost   : ₹{bd.get('total_estimated', 0):,.0f}",
        f"  Remaining        : ₹{bd.get('remaining_budget', 0):,.0f}",
        f"  Status           : {bd.get('budget_status', '')}",
        "",
        "── FLIGHT SUMMARY ─────────────────────────────────────",
        plan.get("flight_summary", ""),
        "",
        "── HOTEL SUMMARY ──────────────────────────────────────",
        plan.get("hotel_summary", ""),
        "",
        "── ITINERARY ───────────────────────────────────────────",
        plan.get("itinerary", ""),
        "",
        "── RECOMMENDATIONS ─────────────────────────────────────",
        plan.get("recommendations", ""),
        "",
        "=" * 60,
        "  Generated by VoyageAI — AI Multi-Agent Travel Planner",
        "=" * 60,
    ]
    return "\n".join(lines)


def _generate_pdf(plan: dict) -> bytes | None:
    """Generate a PDF report using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        import io

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        story  = []

        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            fontSize=22, spaceAfter=6,
            textColor=colors.HexColor("#0d1526"),
        )
        h2_style = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=14, spaceBefore=14, spaceAfter=6,
            textColor=colors.HexColor("#06b6d4"),
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=10, leading=16,
            textColor=colors.HexColor("#1e293b"),
        )

        bd = plan.get("budget_breakdown", {})

        story.append(Paragraph(f"✈ Trip to {plan.get('destination','')}", title_style))
        story.append(Paragraph(
            f"{plan.get('days')} days · {plan.get('travelers')} traveler(s) · ₹{plan.get('budget',0):,.0f} budget",
            body_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
        story.append(Spacer(1, 0.3*cm))

        # Budget table
        story.append(Paragraph("Budget Breakdown", h2_style))
        table_data = [
            ["Category", "Amount (₹)"],
            ["Flights",        f"₹{bd.get('flights',0):,.0f}"],
            ["Accommodation",  f"₹{bd.get('accommodation',0):,.0f}"],
            ["Food",           f"₹{bd.get('food',0):,.0f}"],
            ["Local Transport",f"₹{bd.get('local_transport',0):,.0f}"],
            ["Activities",     f"₹{bd.get('activities',0):,.0f}"],
            ["Miscellaneous",  f"₹{bd.get('miscellaneous',0):,.0f}"],
            ["TOTAL ESTIMATED",f"₹{bd.get('total_estimated',0):,.0f}"],
            ["REMAINING",      f"₹{bd.get('remaining_budget',0):,.0f}"],
        ]
        tbl = Table(table_data, colWidths=[10*cm, 6*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#0d1526")),
            ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS",(0,1), (-1,-2), [colors.HexColor("#f8fafc"), colors.white]),
            ("BACKGROUND",    (0,-2),(-1,-1), colors.HexColor("#dcfce7")),
            ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4*cm))

        for section_title, section_key in [
            ("Flight Recommendations", "flight_summary"),
            ("Hotel Recommendations",  "hotel_summary"),
            ("Day-by-Day Itinerary",   "itinerary"),
            ("Local Recommendations",  "recommendations"),
        ]:
            content = plan.get(section_key, "")
            if content:
                story.append(Paragraph(section_title, h2_style))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
                story.append(Spacer(1, 0.15*cm))
                # Split into paragraphs
                for para in content.split("\n"):
                    if para.strip():
                        try:
                            story.append(Paragraph(para, body_style))
                        except Exception:
                            story.append(Paragraph(para.encode("ascii", "replace").decode(), body_style))
                story.append(Spacer(1, 0.3*cm))

        doc.build(story)
        return buffer.getvalue()

    except ImportError:
        return None
    except Exception:
        return None

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown('<div class="sidebar-brand">✈ VoyageAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">AI-Powered Travel Planning</div>', unsafe_allow_html=True)

    # ── API key check ─────────────────────────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        st.warning("⚠️ Add your GROQ_API_KEY in the .env file to activate AI agents.")
        api_key_input = st.text_input("Or enter API key here:", type="password", key="api_key_input")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            api_key = api_key_input
    st.markdown("---")
    st.markdown("###  Trip Details")

    source = st.text_input(
        " From",
        placeholder="e.g. Delhi, India"
    )

    destination = st.text_input(
        " Destination",
        placeholder="e.g. Tokyo, Japan"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        days = st.number_input(" Days", min_value=1, max_value=30, value=5, step=1)
    with col_b:
        travelers = st.number_input("Travelers", min_value=1, max_value=20, value=2, step=1)

    budget = st.number_input(
        " Budget (₹)",
        min_value=5_000,
        max_value=50_000_000,
        value=100_000,
        step=5_000,
        format="%d",
        help="Total trip budget in Indian Rupees",
    )

    travel_dates = st.text_input(
        " Travel Dates (optional)",
        placeholder="e.g. Dec 2024",
    )

    st.markdown("###  Preferences")
    preferences = st.multiselect(
        "Travel Style",
        options=["Adventure", "Luxury", "Family", "Solo", "Food & Culinary", "Nature", "Shopping", "Culture", "Wellness", "Budget"],
        default=["Food & Culinary"],
        label_visibility="collapsed",
    )

    st.markdown("---")


    col1, col2 = st.columns(2)
    with col1:
        generate_btn = st.button(" Generate Plan", use_container_width=True)
    with col2:
        reset_btn = st.button(" Reset", use_container_width=True)

    if reset_btn:
        st.session_state.plan_result  = None
        st.session_state.is_loading   = False
        st.rerun()

    # ── History ───────────────────────────────────────────────────────────────
    if st.session_state.search_history:
        st.markdown("---")
        st.markdown("###  Recent Searches")
        for i, h in enumerate(reversed(st.session_state.search_history[-5:])):
            if st.button(f" {h['destination']} · {h['days']}d", key=f"hist_{i}", use_container_width=True):
                st.session_state.plan_result = h["result"]
                st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='color:#64748b;font-size:0.72rem;text-align:center;'>"
        "Powered by Groq · Llama 3.3 70B<br>LangGraph Multi-Agent Pipeline"
        "</div>",
        unsafe_allow_html=True,
    )
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN CONTENT                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Hero Banner ───────────────────────────────────────────────────────────────
if not st.session_state.plan_result:
    st.markdown("""
    <div class="hero-banner">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
        <div>
          <div style="font-family:'DM Serif Display',serif; font-size:2.8rem; font-weight:700;
                      background:linear-gradient(135deg,#06b6d4,#8b5cf6,#f59e0b);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            AI Multi-Agent<br>Travel Planner
          </div>
          <div style="color:#94a3b8; font-size:1.05rem; margin-top:0.8rem; max-width:520px;">
            Powered by 5 specialized AI agents working in concert — flights, hotels,
            budget, itinerary, and local recommendations generated in seconds.
          </div>
        </div>
        <div style="display:flex; flex-direction:column; gap:8px; align-self:center;">
          <div class="agent-pill"> Flight Agent</div>
          <div class="agent-pill"> Hotel Agent</div>
          <div class="agent-pill"> Budget Agent</div>
          <div class="agent-pill"> Itinerary Agent</div>
          <div class="agent-pill"> Recommendations</div>
        </div>
      </div>
      <div style="margin-top:2rem; display:flex; gap:2rem; flex-wrap:wrap;">
        <div style="color:#64748b; font-size:0.82rem;">
           <span style="color:#06b6d4">Llama 3.3 70B</span> via Groq
        </div>
        <div style="color:#64748b; font-size:0.82rem;">
           <span style="color:#8b5cf6">LangGraph</span> orchestration
        </div>
        <div style="color:#64748b; font-size:0.82rem;">
           <span style="color:#f59e0b">Guardrails</span> validation
        </div>
        <div style="color:#64748b; font-size:0.82rem;">
           <span style="color:#10b981">Plotly</span> analytics
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown("### How It Works")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1️⃣", "Configure", "Set your destination, budget, dates, and preferences in the sidebar."),
        ("2️⃣", "Validate", "Guardrails agent validates your input and blocks harmful prompts."),
        ("3️⃣", "Agent Pipeline", "5 AI agents run sequentially via LangGraph, each enriching the state."),
        ("4️⃣", "Your Plan", "Receive a full travel plan with analytics, itinerary, and export options."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
              <div style="font-size:2rem;">{icon}</div>
              <div style="font-weight:700; color:#f1f5f9; margin:0.5rem 0;">{title}</div>
              <div style="font-size:0.82rem; color:#94a3b8;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Generate Plan ─────────────────────────────────────────────────────────────
if generate_btn:
    # ── Guardrails ────────────────────────────────────────────────────────────
    result = validate_inputs(
        source=source,
        destination=destination,
        budget=budget,
        days=days,
        travelers=travelers,
        preferences=preferences,
    )

    if not result.valid:
        st.error(f"**Validation Error** — {result.error_message}")
    elif not api_key or api_key == "your_groq_api_key_here":
        st.error(" Please enter a valid GROQ_API_KEY to run the AI agents.")
    else:
        # ── Progress UI ───────────────────────────────────────────────────────
        with st.spinner(""):
            progress_container = st.empty()
            agent_steps = [
                (" Flight Agent",         "Estimating flights & airlines…"),
                (" Hotel Agent",          "Finding accommodation options…"),
                (" Budget Agent",         "Building budget breakdown…"),
                (" Itinerary Agent",      "Crafting day-wise itinerary…"),
                (" Recommendation Agent", "Curating local recommendations…"),
            ]

            for i, (name, desc) in enumerate(agent_steps):
                progress_container.markdown(
                    f"""<div style='text-align:center;padding:2rem;'>
                    <div style='font-size:1.8rem;margin-bottom:1rem;'>🤖</div>
                    <div style='font-family:"DM Serif Display",serif;font-size:1.4rem;color:#f1f5f9;margin-bottom:0.5rem;'>
                      {name}
                    </div>
                    <div style='color:#94a3b8;font-size:0.9rem;margin-bottom:1.5rem;'>{desc}</div>
                    <div style='background:rgba(255,255,255,0.05);border-radius:99px;height:6px;max-width:400px;margin:0 auto;'>
                      <div style='background:linear-gradient(90deg,#06b6d4,#8b5cf6);height:100%;border-radius:99px;
                                  width:{int((i+1)/len(agent_steps)*100)}%;transition:width 0.5s ease;'></div>
                    </div>
                    <div style='color:#64748b;font-size:0.78rem;margin-top:0.8rem;'>
                      Agent {i+1} of {len(agent_steps)}
                    </div></div>""",
                    unsafe_allow_html=True,
                )

            try:
                plan = run_travel_pipeline(
                    source=source,
                    destination=result.sanitized_query or destination,
                    budget=float(budget),
                    days=int(days),
                    travelers=int(travelers),
                    preferences=preferences,
                    travel_dates=travel_dates,
                )
                st.session_state.plan_result = plan

                # Save to history
                st.session_state.search_history.append({
                    "source": source,
                    "destination": destination,
                    "days": days,
                    "budget": budget,
                    "result": plan,
                    "timestamp": datetime.datetime.now().isoformat(),
                })

                # Auto-save JSON
                fname = f"exports/{destination.replace(' ', '_').lower()}_{int(time.time())}.json"
                with open(fname, "w") as f:
                    serializable = {
                        k: v for k, v in plan.items()
                        if isinstance(v, (str, int, float, dict, list, bool, type(None)))
                    }

                    serializable["source"] = source
                    json.dump(serializable, f, indent=2)

            except Exception as e:
                progress_container.empty()
                st.error(f" Agent pipeline error: {str(e)}")
                st.info(" Check your GROQ_API_KEY and network connection.")
                st.stop()

        progress_container.empty()
        st.rerun()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RESULTS DASHBOARD                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

plan = st.session_state.plan_result

if plan:
    bd: dict = plan.get("budget_breakdown", {})
    dest      = plan.get("destination", "Your Destination")
    days_val  = plan.get("days", 1)
    travelers_val = plan.get("travelers", 1)

    total_budget    = bd.get("total_budget", 0)
    total_estimated = bd.get("total_estimated", 0)
    remaining       = bd.get("remaining_budget", 0)
    utilization     = bd.get("budget_utilization", 0)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-banner" style="padding:1.5rem 2rem;">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
        <div>
          <div style="font-family:'DM Serif Display',serif; font-size:1.8rem; color:#f1f5f9;">
            Your Trip to <span style="color:#06b6d4;">{dest}</span>
          </div>
          <div style="color:#94a3b8; font-size:0.88rem; margin-top:4px;">
            {days_val} days · {travelers_val} traveler(s) · Generated {datetime.datetime.now().strftime('%B %d, %Y')}
          </div>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <span class="agent-pill done">✅ Plan Complete</span>
          <span class="agent-pill done">5 Agents Executed</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Total Budget",    f"₹{total_budget:,.0f}",    "Your planned spend",      "#f59e0b", "linear-gradient(135deg,#f59e0b,#f97316)"),
        ("Estimated Cost",  f"₹{total_estimated:,.0f}", "All expenses combined",   "#06b6d4", "linear-gradient(135deg,#06b6d4,#0ea5e9)"),
        ("Potential Savings", f"₹{max(remaining,0):,.0f}", bd.get("budget_status", ""), "#10b981", "linear-gradient(135deg,#10b981,#059669)"),
        ("Trip Score",      f"{max(0,100-int(utilization))+50}/100", f"{utilization:.0f}% utilized", "#8b5cf6", "linear-gradient(135deg,#8b5cf6,#7c3aed)"),
    ]
    for col, (label, value, sub, color, grad) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="glow-bar" style="background:{grad};"></div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color};">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Main Tabs ─────────────────────────────────────────────────────────────
    tabs = st.tabs([" Overview", " Flights", " Hotels", " Itinerary", " Budget Analytics", " Recommendations", " Export"])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="section-title"> Trip Overview</div>', unsafe_allow_html=True)

        oc1, oc2 = st.columns([2, 1])
        with oc1:
            flight_cost  = bd.get("flights", 0)
            hotel_cost   = bd.get("accommodation", 0)
            food_cost    = bd.get("food", 0)
            trans_cost   = bd.get("local_transport", 0)
            act_cost     = bd.get("activities", 0)
            misc_cost    = bd.get("miscellaneous", 0)

            # Mini budget bar
            categories = [
                (" Flights",        flight_cost,  "#06b6d4"),
                (" Accommodation",  hotel_cost,   "#8b5cf6"),
                (" Food",          food_cost,    "#f59e0b"),
                (" Transport",      trans_cost,   "#10b981"),
                (" Activities",     act_cost,     "#f43f5e"),
                (" Misc",           misc_cost,    "#94a3b8"),
            ]
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("** Budget at a Glance**")
            for cat, amt, color in categories:
                pct = (amt / total_budget * 100) if total_budget else 0
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:10px; margin:6px 0;">
                  <div style="width:120px; font-size:0.82rem; color:#94a3b8;">{cat}</div>
                  <div style="flex:1; background:rgba(255,255,255,0.05); border-radius:99px; height:8px; overflow:hidden;">
                    <div style="width:{pct:.1f}%; background:{color}; height:100%; border-radius:99px;"></div>
                  </div>
                  <div style="width:90px; text-align:right; font-size:0.82rem; color:#f1f5f9;">₹{amt:,.0f}</div>
                  <div style="width:40px; text-align:right; font-size:0.75rem; color:#64748b;">{pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with oc2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("** Quick Stats**")
            fd = plan.get("flight_data", {})
            hd = plan.get("hotel_data",  {})
            stats = [
                ("Daily Avg Spend",  f"₹{bd.get('daily_average',0):,}"),
                ("Per Person Total", f"₹{int(total_estimated/max(travelers_val,1)):,}"),
                ("Flight Duration",  fd.get("duration_hours", "–") + " hrs"),
                ("Hotel Nights",     f"{hd.get('nights', days_val)} nights"),
                ("Budget Status",    bd.get("budget_status", "–")),
            ]
            for label, val in stats:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:7px 0;
                            border-bottom:1px solid rgba(255,255,255,0.05); font-size:0.85rem;">
                  <span style="color:#94a3b8;">{label}</span>
                  <span style="color:#f1f5f9; font-weight:600;">{val}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Weather
            wt = fd.get("best_travel_months", "")
            if wt:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("** Best Time to Visit**")
                st.markdown(f"<span style='color:#f59e0b;font-size:0.9rem;'>{wt}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: Flights ────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="section-title"> Flight Recommendations</div>', unsafe_allow_html=True)

        fd = plan.get("flight_data", {})
        if fd:
            fc1, fc2, fc3 = st.columns(3)
            metrics = [
                ("Cost Per Person", f"₹{fd.get('cost_per_person_inr',0):,.0f}", "Round trip"),
                ("Total Flight Cost", f"₹{fd.get('total_cost_inr',0):,.0f}", f"{travelers_val} traveler(s)"),
                ("Flight Duration", f"{fd.get('duration_hours','–')} hrs", "Approx."),
            ]
            for col, (label, val, sub) in zip([fc1, fc2, fc3], metrics):
                with col:
                    st.metric(label, val, sub)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("** Recommended Airlines**")
            for airline in fd.get("airlines", []):
                st.markdown(f"• **{airline}**", unsafe_allow_html=False)
            st.markdown(f"\n **{fd.get('recommended_booking','')}**")
            st.markdown(f"\n Best Travel Months: **{fd.get('best_travel_months','')}**")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("###  AI Flight Analysis")
        st.markdown(
            f'<div class="glass-card">{plan.get("flight_summary","")}</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 3: Hotels ─────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-title"> Accommodation Options</div>', unsafe_allow_html=True)

        hd = plan.get("hotel_data", {})
        if hd:
            options = hd.get("options", [])
            tier_colors = {"Budget": "#10b981", "Mid-range": "#06b6d4", "Luxury": "#f59e0b"}
            h_cols = st.columns(len(options)) if options else []
            for col, opt in zip(h_cols, options):
                with col:
                    tier  = opt.get("type", "")
                    color = tier_colors.get(tier, "#94a3b8")
                    st.markdown(f"""
                    <div class="glass-card" style="border-top:3px solid {color};">
                      <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                                  letter-spacing:1px; color:{color}; margin-bottom:0.5rem;">{tier}</div>
                      <div style="font-size:1rem; font-weight:700; color:#f1f5f9; margin-bottom:0.8rem;">
                        {opt.get("name","")}
                      </div>
                      <div style="font-size:1.6rem; font-weight:700; color:{color}; font-family:'DM Serif Display',serif;">
                        ₹{opt.get("price",0):,}
                      </div>
                      <div style="font-size:0.8rem; color:#64748b;">per night</div>
                      <div style="margin-top:0.8rem; font-size:0.85rem; color:#f59e0b;">
                        {'⭐' * int(opt.get('rating',3))} {opt.get('rating','')}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            rec = hd.get("recommended", {})
            if rec:
                st.info(f" **Recommended:** {rec.get('name','')} — ₹{rec.get('price',0):,}/night  •  Total: ₹{hd.get('total_hotel_cost',0):,}")

        st.markdown("###  AI Hotel Analysis")
        st.markdown(
            f'<div class="glass-card">{plan.get("hotel_summary","")}</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 4: Itinerary ──────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="section-title"> Day-by-Day Itinerary</div>', unsafe_allow_html=True)

        itinerary_text = plan.get("itinerary", "")
        if itinerary_text:
            # Parse days and render as cards
            lines = itinerary_text.split("\n")
            current_day_lines: list[str] = []
            day_blocks: list[list[str]] = []

            for line in lines:
                if line.strip().lower().startswith("day ") and current_day_lines:
                    day_blocks.append(current_day_lines)
                    current_day_lines = [line]
                elif line.strip():
                    current_day_lines.append(line)
            if current_day_lines:
                day_blocks.append(current_day_lines)

            colors = ["#06b6d4", "#8b5cf6", "#f59e0b", "#10b981", "#f43f5e",
                      "#06b6d4", "#8b5cf6", "#f59e0b", "#10b981", "#f43f5e"]

            for i, block in enumerate(day_blocks):
                color = colors[i % len(colors)]
                header = block[0] if block else f"Day {i+1}"
                body   = "\n".join(block[1:]) if len(block) > 1 else ""
                st.markdown(f"""
                <div style="border-left:3px solid {color}; border-radius:0 14px 14px 0;
                            background:rgba(15,23,42,0.6); padding:1rem 1.5rem; margin-bottom:1rem;">
                  <div style="color:{color}; font-weight:700; font-size:1rem; margin-bottom:0.5rem;">{header}</div>
                  <div style="color:#cbd5e1; font-size:0.88rem; line-height:1.7; white-space:pre-line;">{body}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Itinerary not available.")

    # ── Tab 5: Budget Analytics ───────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-title"> Budget Analytics</div>', unsafe_allow_html=True)

        # Chart data
        labels = ["Flights", "Accommodation", "Food", "Transport", "Activities", "Misc"]
        values = [
            bd.get("flights", 0),
            bd.get("accommodation", 0),
            bd.get("food", 0),
            bd.get("local_transport", 0),
            bd.get("activities", 0),
            bd.get("miscellaneous", 0),
        ]
        colors_chart = ["#06b6d4", "#8b5cf6", "#f59e0b", "#10b981", "#f43f5e", "#94a3b8"]

        _transparent = "rgba(0,0,0,0)"
        chart_layout = dict(
            paper_bgcolor=_transparent,
            plot_bgcolor=_transparent,
            font=dict(color="#94a3b8", family="Outfit"),
            margin=dict(t=40, b=20, l=20, r=20),
        )

        ch1, ch2 = st.columns(2)

        # ── Pie chart ─────────────────────────────────────────────────────────
        with ch1:
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker=dict(colors=colors_chart, line=dict(color="#080c14", width=2)),
                textfont=dict(size=12, color="#f1f5f9"),
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                **chart_layout,
                title=dict(text="Budget Breakdown", font=dict(size=16, color="#f1f5f9")),
                showlegend=True,
                legend=dict(
                    orientation="v", font=dict(color="#94a3b8"),
                    bgcolor=_transparent,
                ),
                annotations=[dict(
                    text=f"₹{total_estimated:,.0f}", x=0.5, y=0.5,
                    font=dict(size=14, color="#f1f5f9", family="DM Serif Display"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Bar chart ─────────────────────────────────────────────────────────
        with ch2:
            # Daily spend simulation
            daily_labels = [f"Day {i+1}" for i in range(days_val)]
            base_daily = bd.get("daily_average", 2000)
            import random; random.seed(42)
            daily_vals = [round(base_daily * random.uniform(0.8, 1.3)) for _ in range(days_val)]

            fig_bar = go.Figure(go.Bar(
                x=daily_labels, y=daily_vals,
                marker=dict(
                    color=daily_vals,
                    colorscale=[[0, "#06b6d4"], [0.5, "#8b5cf6"], [1, "#f59e0b"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
            ))
            fig_bar.update_layout(
                **chart_layout,
                title=dict(text="Estimated Daily Spend", font=dict(size=16, color="#f1f5f9")),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8"),
                           tickprefix="₹"),
                bargap=0.3,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        ch3, ch4 = st.columns(2)

        # ── Sunburst ──────────────────────────────────────────────────────────
        with ch3:
            sub_labels = (
                ["Total Trip"] + labels +
                ["Flights-detail", "Hotels-detail", "Food-detail",
                 "Transport-detail", "Activities-detail", "Misc-detail"]
            )
            sub_parents = (
                [""] + ["Total Trip"] * 6 +
                ["Flights", "Accommodation", "Food", "Transport", "Activities", "Misc"]
            )
            sub_values = (
                [total_estimated] + values +
                [v * 0.6 for v in values]
            )
            fig_sun = go.Figure(go.Sunburst(
                labels=sub_labels, parents=sub_parents, values=sub_values,
                branchvalues="total",
                marker=dict(colors=["#0d1526"] + colors_chart + [c + "88" for c in colors_chart]),
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>",
                textfont=dict(color="#f1f5f9"),
            ))
            fig_sun.update_layout(
                **chart_layout,
                title=dict(text="Cost Distribution", font=dict(size=16, color="#f1f5f9")),
            )
            st.plotly_chart(fig_sun, use_container_width=True)

        # ── Gauge ─────────────────────────────────────────────────────────────
        with ch4:
            gauge_val = min(utilization, 100)
            gauge_color = "#10b981" if gauge_val < 70 else "#f59e0b" if gauge_val < 90 else "#f43f5e"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=gauge_val,
                delta={"reference": 80, "valueformat": ".1f",
                       "increasing": {"color": "#f43f5e"}, "decreasing": {"color": "#10b981"}},
                number={"suffix": "%", "font": {"color": gauge_color, "size": 36, "family": "DM Serif Display"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#94a3b8"}, "tickwidth": 1,
                             "tickcolor": "rgba(255,255,255,0.1)"},
                    "bar": {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "rgba(255,255,255,0.03)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 70],  "color": "rgba(16,185,129,0.1)"},
                        {"range": [70, 90], "color": "rgba(245,158,11,0.1)"},
                        {"range": [90,100], "color": "rgba(244,63,94,0.1)"},
                    ],
                    "threshold": {"line": {"color": "#f1f5f9", "width": 2}, "value": 80},
                },
            ))
            fig_gauge.update_layout(
                **chart_layout,
                title=dict(text="Budget Utilization", font=dict(size=16, color="#f1f5f9")),
                height=280,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("###  AI Budget Analysis")
        st.markdown(
            f'<div class="glass-card">{plan.get("budget_summary","")}</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 6: Recommendations ────────────────────────────────────────────────
    with tabs[5]:
        st.markdown('<div class="section-title">⭐ Local Recommendations</div>', unsafe_allow_html=True)
        recs = plan.get("recommendations", "")
        if recs:
            # Section-aware rendering
            section_icons = {
                "🍽️": "#f59e0b",
                "🏛️": "#06b6d4",
                "💎": "#8b5cf6",
                "💡": "#10b981",
                "🛍️": "#f43f5e",
            }
            current_section: list[str] = []
            sections: list[tuple[str, str, list[str]]] = []
            current_icon = "⭐"
            current_color = "#94a3b8"

            for line in recs.split("\n"):
                found = False
                for icon, color in section_icons.items():
                    if icon in line:
                        if current_section:
                            sections.append((current_icon, current_color, current_section))
                        current_section = [line]
                        current_icon    = icon
                        current_color   = color
                        found = True
                        break
                if not found and line.strip():
                    current_section.append(line)
            if current_section:
                sections.append((current_icon, current_color, current_section))

            if sections:
                for icon, color, sec_lines in sections:
                    header = sec_lines[0] if sec_lines else ""
                    body   = "\n".join(sec_lines[1:]) if len(sec_lines) > 1 else ""
                    st.markdown(f"""
                    <div class="glass-card" style="border-left:3px solid {color};">
                      <div style="color:{color}; font-weight:700; font-size:1rem; margin-bottom:0.8rem;">{header}</div>
                      <div style="color:#cbd5e1; font-size:0.87rem; line-height:1.75; white-space:pre-line;">{body}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="glass-card"><div style="white-space:pre-line;line-height:1.75;">{recs}</div></div>',
                    unsafe_allow_html=True,
                )

    # ── Tab 7: Export ─────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="section-title"> Export Your Plan</div>', unsafe_allow_html=True)

        ex1, ex2 = st.columns(2)

        with ex1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("###  Export as JSON")
            serializable = {
                k: v for k, v in plan.items()
                if isinstance(v, (str, int, float, dict, list, bool, type(None)))
            }
            json_str = json.dumps(serializable, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"trip_{dest.replace(' ','_').lower()}.json",
                mime="application/json",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with ex2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("###  Export as Text")
            text_report = _build_text_report(plan)
            st.download_button(
                label="⬇️ Download TXT",
                data=text_report,
                file_name=f"trip_{dest.replace(' ','_').lower()}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # PDF export
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Export as PDF")
        if st.button("Generate & Download PDF", use_container_width=True):
            with st.spinner("Generating PDF…"):
                pdf_bytes = _generate_pdf(plan)
                if pdf_bytes:
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"trip_{dest.replace(' ','_').lower()}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.warning("PDF generation failed. Install reportlab: `pip install reportlab`")
        st.markdown("</div>", unsafe_allow_html=True)

        # Raw JSON viewer
        with st.expander(" View Raw Plan Data"):
            st.json(serializable)
    