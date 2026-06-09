"""
agents/recommendation_agent.py
--------------------------------
Recommendation Agent: curates restaurants, attractions,
hidden gems, shopping spots, and travel tips.
"""

from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """
You are a seasoned travel expert, local guide, and travel blogger.

Responsibilities:

1. Recommend authentic restaurants
2. Suggest must-visit attractions
3. Discover hidden gems
4. Recommend shopping areas
5. Provide practical travel advice

Recommendations should:

- Match traveler interests
- Respect budget constraints
- Avoid tourist traps when possible
- Include local favorites
- Be realistic and useful

Structure response exactly as:

🍽️ Restaurants

🏛️ Attractions

💎 Hidden Gems

🛍️ Shopping

💡 Travel Tips
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
    Determine currency from source country.
    """

    source = source.lower()

    for country, currency in COUNTRY_CURRENCY.items():
        if country in source:
            return currency

    return "USD"


# ------------------------------------------------------------------
# Recommendation Agent
# ------------------------------------------------------------------

def run_recommendation_agent(
    state: dict[str, Any],
    llm: ChatGroq,
) -> dict[str, Any]:
    """
    Execute Recommendation Agent node.
    """

    source = state["source"]
    destination = state["destination"]

    days = state["days"]
    travelers = state["travelers"]

    preferences = state.get("preferences", [])

    currency = get_currency(source)

    hotel_data = state.get("hotel_data", {})
    budget_data = state.get("budget_breakdown", {})
    itinerary = state.get("itinerary", "")

    accommodation_area = hotel_data.get(
        "recommended_area",
        "city center",
    )

    daily_budget = budget_data.get(
        "daily_average",
        "N/A",
    )

    budget_status = budget_data.get(
        "budget_status",
        "On Budget",
    )

    pref_str = (
        ", ".join(preferences)
        if preferences
        else "General Travel"
    )

    user_msg = f"""
Create personalized travel recommendations.

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

Currency:
{currency}

Budget Status:
{budget_status}

Daily Budget:
{daily_budget} {currency}

Accommodation Area:
{accommodation_area}

Planned Itinerary:
{itinerary[:2000]}

Provide:

🍽️ TOP RESTAURANTS (5)

For each:
- Name
- Cuisine
- Approximate cost per person ({currency})
- Signature dish

🏛️ TOP ATTRACTIONS (5)

For each:
- Name
- Why visit
- Best visiting time
- Entry fee ({currency} if applicable)

💎 HIDDEN GEMS (3)

For each:
- Why locals love it
- Best time to visit

🛍️ SHOPPING SPOTS (3)

For each:
- What to buy
- Price expectations
- Bargaining advice if applicable

💡 INSIDER TRAVEL TIPS (5)

Include:
- Cultural etiquette
- Transport advice
- Safety tips
- Scam warnings
- Money-saving advice

Requirements:

- Match traveler interests
- Stay aligned with budget
- Avoid generic suggestions
- Prefer authentic local experiences

Keep each recommendation concise.
Total response under 500 words.
"""

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    )

    return {
        **state,
        "recommendations": response.content,
    }