"""
agents/itinerary_agent.py
--------------------------
Itinerary Agent: generates a detailed day-wise travel plan.
"""

from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """
You are an expert travel itinerary designer.

Responsibilities:

1. Create realistic daily itineraries
2. Balance sightseeing and relaxation
3. Consider traveler preferences
4. Respect budget constraints
5. Minimize unnecessary travel time

For each day include:

- Morning
- Lunch
- Afternoon
- Evening
- Daily estimated spend
- Local tip

Format:

Day 1: Theme
Day 2: Theme

Use concise bullet points.
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
    source = source.lower()

    for country, currency in COUNTRY_CURRENCY.items():
        if country in source:
            return currency

    return "USD"


# ------------------------------------------------------------------
# Itinerary Agent
# ------------------------------------------------------------------

def run_itinerary_agent(
    state: dict[str, Any],
    llm: ChatGroq,
) -> dict[str, Any]:
    """
    Generate a detailed itinerary.
    """

    source = state["source"]
    destination = state["destination"]

    days = state["days"]
    travelers = state["travelers"]
    budget = state["budget"]

    preferences = state.get("preferences", [])

    currency = get_currency(source)

    flight_data = state.get("flight_data", {})
    hotel_data = state.get("hotel_data", {})
    budget_data = state.get("budget_breakdown", {})

    accommodation_area = hotel_data.get(
        "recommended_area",
        "city center",
    )

    daily_budget = budget_data.get(
        "daily_average",
        round(budget / max(days, 1)),
    )

    budget_status = budget_data.get(
        "budget_status",
        "On Budget",
    )

    pref_str = (
        ", ".join(preferences)
        if preferences
        else "General sightseeing"
    )

    user_msg = f"""
Create a detailed travel itinerary.

Source:
{source}

Destination:
{destination}

Trip Duration:
{days} days

Travelers:
{travelers}

Preferences:
{pref_str}

Budget:
{budget} {currency}

Budget Status:
{budget_status}

Recommended Accommodation Area:
{accommodation_area}

Daily Budget:
{daily_budget} {currency}

Flight Information:
{flight_data}

For EACH day provide:

1. Theme title
2. Morning activity
3. Lunch recommendation
4. Afternoon activity
5. Evening activity
6. Daily estimated spend in {currency}
7. Local insider tip

Requirements:

- Stay within budget
- Include famous attractions
- Include hidden gems
- Minimize travel time
- Use realistic timings
- Mention transportation where useful

Keep each day around 120-150 words.
"""

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    )

    return {
        **state,
        "itinerary": response.content,
    }