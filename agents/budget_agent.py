"""
agents/budget_agent.py
-----------------------
Budget Agent: consolidates all costs and produces a budget breakdown.
"""

from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """
You are a certified travel budget advisor.

Analyze travel expenses and provide:

1. Budget health analysis
2. Cost optimization suggestions
3. Areas to save money
4. Areas worth spending more
5. Daily spending recommendation

Keep response under 250 words.
"""


# ---------------------------------------------------------------------
# Currency Detection
# ---------------------------------------------------------------------

COUNTRY_CURRENCY = {
    "india": "INR",
    "usa": "USD",
    "united states": "USD",
    "uk": "GBP",
    "united kingdom": "GBP",
    "england": "GBP",
    "canada": "CAD",
    "australia": "AUD",
    "japan": "JPY",
    "singapore": "SGD",
    "uae": "AED",
    "dubai": "AED",
    "france": "EUR",
    "germany": "EUR",
    "italy": "EUR",
    "spain": "EUR",
    "netherlands": "EUR",
}


def get_currency(source: str) -> str:
    """
    Determine currency from source country.
    """

    src = source.lower()

    for country, currency in COUNTRY_CURRENCY.items():
        if country in src:
            return currency

    return "USD"


# ---------------------------------------------------------------------
# Cost Heuristics
# ---------------------------------------------------------------------

def estimate_food_cost(
    destination: str,
    travelers: int,
    preferences: list[str],
) -> float:

    base = 20

    destination = destination.lower()

    multipliers = {
        "japan": 2.0,
        "usa": 2.5,
        "london": 2.7,
        "paris": 2.5,
        "dubai": 2.2,
        "singapore": 2.0,
        "thailand": 1.2,
        "bali": 1.0,
        "malaysia": 1.1,
        "vietnam": 0.9,
        "goa": 0.8,
    }

    mult = 1.5

    for k, v in multipliers.items():
        if k in destination:
            mult = v
            break

    if "luxury" in [p.lower() for p in preferences]:
        mult *= 1.5

    return round(base * mult * travelers)


def estimate_transport_cost(
    destination: str,
    travelers: int,
) -> float:

    destination = destination.lower()

    mapping = {
        "japan": 15,
        "usa": 25,
        "dubai": 12,
        "singapore": 10,
        "thailand": 8,
        "bali": 7,
        "goa": 5,
    }

    base = 10

    for k, v in mapping.items():
        if k in destination:
            base = v
            break

    return round(base * travelers)


def estimate_activity_cost(
    destination: str,
    travelers: int,
    preferences: list[str],
) -> float:

    base = 20

    destination = destination.lower()

    multipliers = {
        "japan": 2.0,
        "usa": 2.5,
        "dubai": 3.0,
        "singapore": 2.0,
        "thailand": 1.2,
        "bali": 1.0,
        "maldives": 4.0,
    }

    mult = 1.5

    for k, v in multipliers.items():
        if k in destination:
            mult = v
            break

    prefs = [p.lower() for p in preferences]

    if "adventure" in prefs:
        mult *= 1.3

    if "luxury" in prefs:
        mult *= 1.8

    return round(base * mult * travelers)


# ---------------------------------------------------------------------
# Budget Agent
# ---------------------------------------------------------------------

def run_budget_agent(
    state: dict[str, Any],
    llm: ChatGroq,
) -> dict[str, Any]:

    source = state["source"]
    destination = state["destination"]
    budget = state["budget"]
    days = state["days"]
    travelers = state["travelers"]
    preferences = state.get("preferences", [])

    currency = get_currency(source)

    flight_cost = (
        state.get("flight_data", {})
        .get("total_cost", 0)
    )

    hotel_cost = (
        state.get("hotel_data", {})
        .get("total_hotel_cost", 0)
    )

    food_per_day = estimate_food_cost(
        destination,
        travelers,
        preferences,
    )

    transport_per_day = estimate_transport_cost(
        destination,
        travelers,
    )

    activities_per_day = estimate_activity_cost(
        destination,
        travelers,
        preferences,
    )

    total_food = food_per_day * days
    total_transport = transport_per_day * days
    total_activities = activities_per_day * days

    misc = budget * 0.05

    total_estimated = (
        flight_cost
        + hotel_cost
        + total_food
        + total_transport
        + total_activities
        + misc
    )

    remaining = budget - total_estimated

    if remaining >= budget * 0.15:
        status = "✅ Under Budget"
    elif remaining >= 0:
        status = "⚠️ On Budget"
    else:
        status = "❌ Over Budget"

    breakdown = {
        "currency": currency,
        "total_budget": round(budget),
        "flights": round(flight_cost),
        "accommodation": round(hotel_cost),
        "food": round(total_food),
        "local_transport": round(total_transport),
        "activities": round(total_activities),
        "miscellaneous": round(misc),
        "total_estimated": round(total_estimated),
        "remaining_budget": round(remaining),
        "budget_status": status,
        "budget_utilization": round(
            (total_estimated / max(budget, 1)) * 100,
            1,
        ),
        "daily_average": round(
            total_estimated / max(days, 1)
        ),
    }

    user_msg = f"""
Source: {source}
Destination: {destination}

Travelers: {travelers}
Days: {days}

Budget: {budget} {currency}

Flights: {flight_cost}
Accommodation: {hotel_cost}
Food: {total_food}
Transport: {total_transport}
Activities: {total_activities}
Misc: {misc}

Total Estimated: {total_estimated}

Budget Status:
{status}
"""

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    )

    return {
        **state,
        "budget_breakdown": breakdown,
        "budget_summary": response.content,
    }