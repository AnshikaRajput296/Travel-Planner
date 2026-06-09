"""
agents/hotel_agent.py
----------------------
Hotel Agent: suggests accommodations across budget tiers.
"""

from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from tools.hotel_tool import search_hotels


SYSTEM_PROMPT = """
You are a professional travel accommodation consultant.

Responsibilities:

1. Recommend hotels for different budgets
2. Suggest suitable neighborhoods
3. Explain value for money
4. Highlight amenities
5. Mention booking recommendations

Always provide:

- Budget option
- Mid-range option
- Luxury option

Keep response under 300 words.
"""


# ------------------------------------------------------------------
# Currency Detection
# ------------------------------------------------------------------

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
    Detect currency from source country.
    """

    source = source.lower()

    for country, currency in COUNTRY_CURRENCY.items():
        if country in source:
            return currency

    return "USD"


# ------------------------------------------------------------------
# Hotel Agent
# ------------------------------------------------------------------

def run_hotel_agent(
    state: dict[str, Any],
    llm: ChatGroq,
) -> dict[str, Any]:
    """
    Execute Hotel Agent node.

    Returns:
        hotel_data
        hotel_summary
    """

    source = state["source"]
    destination = state["destination"]
    budget = state["budget"]
    days = state["days"]
    travelers = state["travelers"]
    preferences = state.get("preferences", [])

    currency = get_currency(source)

    # --------------------------------------------------------------
    # Get flight cost from Flight Agent
    # --------------------------------------------------------------

    flight_cost = (
        state.get("flight_data", {})
        .get("total_cost", 0)
    )

    remaining_budget = max(
        budget - flight_cost,
        budget * 0.20,
    )

    budget_per_night = round(
        (remaining_budget * 0.35) / max(days, 1)
    )

    # --------------------------------------------------------------
    # Hotel Tool
    # --------------------------------------------------------------

    raw_hotel_info = search_hotels.invoke(
        {
            "destination": destination,
            "budget_per_night": budget_per_night,
            "days": days,
            "travelers": travelers,
        }
    )

    # --------------------------------------------------------------
    # Normalize Tool Output
    # --------------------------------------------------------------

    options = raw_hotel_info.get(
        "options",
        [],
    )

    total_hotel_cost = raw_hotel_info.get(
        "total_hotel_cost",
        budget_per_night * days,
    )

    recommended_area = raw_hotel_info.get(
        "recommended_area",
        "City Center",
    )

    booking_platforms = raw_hotel_info.get(
        "booking_platforms",
        [
            "Booking.com",
            "Agoda",
            "Hotels.com",
        ],
    )

    hotel_data = {
        "source": source,
        "destination": destination,
        "currency": currency,
        "budget_per_night": budget_per_night,
        "total_hotel_cost": round(total_hotel_cost),
        "recommended_area": recommended_area,
        "booking_platforms": booking_platforms,
        "options": options,
    }

    # --------------------------------------------------------------
    # LLM Recommendation
    # --------------------------------------------------------------

    user_msg = f"""
Source:
{source}

Destination:
{destination}

Trip Length:
{days} days

Travelers:
{travelers}

Preferences:
{', '.join(preferences) if preferences else 'General'}

Currency:
{currency}

Estimated Hotel Budget:
{budget_per_night} {currency} per night

Hotel Data:
{hotel_data}

Provide:

1. Best budget hotel option
2. Best mid-range hotel option
3. Best luxury hotel option
4. Best neighborhood to stay
5. Booking platform recommendation
6. Cancellation and booking tips

Keep response under 300 words.
"""

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    )

    return {
        **state,
        "hotel_data": hotel_data,
        "hotel_summary": response.content,
    }