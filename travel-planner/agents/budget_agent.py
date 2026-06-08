"""
agents/budget_agent.py
-----------------------
Budget Agent: consolidates all costs and produces a budget breakdown.
"""

from __future__ import annotations
from typing import Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are a certified financial travel advisor.
Your specialty is creating detailed, realistic travel budgets.
You break costs into: flights, accommodation, food, local transport, activities, shopping, and miscellaneous.
Always provide a clear budget health status: Under Budget / On Budget / Over Budget."""


def run_budget_agent(state: dict[str, Any], llm: ChatGroq) -> dict[str, Any]:
    """
    Execute the Budget Agent node.

    Aggregates flight and hotel costs, estimates remaining categories,
    and produces a full budget breakdown.
    """
    destination = state["destination"]
    budget      = state["budget"]
    days        = state["days"]
    travelers   = state["travelers"]
    preferences = state.get("preferences", [])

    flight_cost = state.get("flight_data", {}).get("total_cost_inr", 0)
    hotel_cost  = state.get("hotel_data",  {}).get("total_hotel_cost", 0)

    # Heuristic daily estimates per person in INR
    food_per_day        = _food_cost(destination, preferences) * travelers
    transport_per_day   = _transport_cost(destination) * travelers
    activities_per_day  = _activity_cost(destination, preferences) * travelers

    total_food       = food_per_day * days
    total_transport  = transport_per_day * days
    total_activities = activities_per_day * days
    misc             = budget * 0.05  # 5 % buffer

    total_estimated = flight_cost + hotel_cost + total_food + total_transport + total_activities + misc
    remaining       = budget - total_estimated

    if remaining >= budget * 0.15:
        status = "✅ Under Budget"
    elif remaining >= 0:
        status = "⚠️ On Budget"
    else:
        status = "❌ Over Budget"

    breakdown = {
        "total_budget":         budget,
        "flights":              flight_cost,
        "accommodation":        hotel_cost,
        "food":                 total_food,
        "local_transport":      total_transport,
        "activities":           total_activities,
        "miscellaneous":        round(misc),
        "total_estimated":      round(total_estimated),
        "remaining_budget":     round(remaining),
        "budget_status":        status,
        "budget_utilization":   round((total_estimated / budget) * 100, 1),
        "daily_average":        round(total_estimated / max(days, 1)),
    }

    user_msg = (
        f"Destination: {destination} | {days} days | {travelers} traveler(s)\n"
        f"Total Budget: ₹{budget:,.0f}\n"
        f"Breakdown:\n"
        f"  Flights:        ₹{flight_cost:,.0f}\n"
        f"  Accommodation:  ₹{hotel_cost:,.0f}\n"
        f"  Food:           ₹{total_food:,.0f}\n"
        f"  Transport:      ₹{total_transport:,.0f}\n"
        f"  Activities:     ₹{total_activities:,.0f}\n"
        f"  Misc:           ₹{misc:,.0f}\n"
        f"  Total:          ₹{total_estimated:,.0f}\n"
        f"  Status:         {status}\n\n"
        "Provide:\n"
        "1. Budget health analysis\n"
        "2. Top 3 money-saving tips for this destination\n"
        "3. Where to splurge vs. save\n"
        "4. Daily budget suggestion per person\n"
        "Keep under 250 words."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "budget_breakdown": breakdown,
        "budget_summary": response.content,
    }


# ── Cost heuristics ────────────────────────────────────────────────────────

def _food_cost(destination: str, preferences: list) -> int:
    """Return daily food cost per person in INR."""
    base = 800
    d = destination.lower()
    mults = {"japan": 4, "usa": 5, "europe": 4.5, "london": 5, "paris": 5,
             "dubai": 3.5, "singapore": 3, "australia": 5, "maldives": 6,
             "thailand": 1.8, "bali": 2, "malaysia": 1.8, "vietnam": 1.5,
             "goa": 1.2, "kerala": 1.1}
    mult = 2.5
    for key, val in mults.items():
        if key in d:
            mult = val
            break
    if "luxury" in [p.lower() for p in preferences]:
        mult *= 1.5
    return round(base * mult)


def _transport_cost(destination: str) -> int:
    """Return daily local transport cost per person in INR."""
    d = destination.lower()
    map_ = {"japan": 1500, "usa": 2500, "europe": 2000, "dubai": 1000,
            "singapore": 800, "thailand": 600, "bali": 500, "maldives": 2000,
            "australia": 2000, "goa": 400, "kerala": 350}
    for key, val in map_.items():
        if key in d:
            return val
    return 1000


def _activity_cost(destination: str, preferences: list) -> int:
    """Return daily activities cost per person in INR."""
    base = 800
    d = destination.lower()
    mults = {"japan": 3, "usa": 4, "europe": 3.5, "dubai": 4,
             "maldives": 8, "bali": 2.5, "thailand": 2, "singapore": 3}
    mult = 2.0
    for key, val in mults.items():
        if key in d:
            mult = val
            break
    if "adventure" in [p.lower() for p in preferences]:
        mult *= 1.3
    if "luxury" in [p.lower() for p in preferences]:
        mult *= 1.8
    return round(base * mult)
