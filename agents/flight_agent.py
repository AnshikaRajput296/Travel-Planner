"""
agents/flight_agent.py
-----------------------
Flight Agent: estimates costs, suggests airlines, recommends travel times.
"""

from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from tools.flight_tool import search_flights


SYSTEM_PROMPT = """
You are an expert flight booking assistant.

Your responsibilities:

1. Analyze flight options
2. Recommend airlines
3. Estimate realistic round-trip airfare
4. Suggest booking windows
5. Provide travel tips

Always:
- Use the currency of the source country
- Mention airline recommendations
- Mention layovers and duration
- Mention baggage considerations
- Keep response under 300 words
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
# Flight Agent
# ------------------------------------------------------------------

def run_flight_agent(
    state: dict[str, Any],
    llm: ChatGroq,
) -> dict[str, Any]:
    """
    Execute Flight Agent node.

    Inputs:
        source
        destination
        budget
        travelers
        days

    Outputs:
        flight_data
        flight_summary
    """

    source = state["source"]
    destination = state["destination"]
    budget = state["budget"]
    travelers = state["travelers"]
    days = state["days"]

    currency = get_currency(source)

    # --------------------------------------------------------------
    # Flight Search Tool
    # --------------------------------------------------------------

    raw_flight_info = search_flights.invoke(
        {
            "source": source,
            "destination": destination,
            "budget": budget,
            "travelers": travelers,
        }
    )

    # --------------------------------------------------------------
    # Normalize Tool Output
    # --------------------------------------------------------------

    airlines = raw_flight_info.get(
        "airlines",
        [
            "Singapore Airlines",
            "Qatar Airways",
            "Emirates",
        ],
    )

    estimated_cost = raw_flight_info.get(
        "total_cost",
        raw_flight_info.get(
            "estimated_cost",
            budget * 0.35,
        ),
    )

    duration = raw_flight_info.get(
        "duration",
        "8-12 hours",
    )

    layovers = raw_flight_info.get(
        "layovers",
        "1 stop",
    )

    booking_window = raw_flight_info.get(
        "booking_window",
        "6-8 weeks before departure",
    )

    flight_data = {
        "source": source,
        "destination": destination,
        "currency": currency,
        "airlines": airlines,
        "total_cost": round(float(estimated_cost)),
        "duration": duration,
        "layovers": layovers,
        "booking_window": booking_window,
    }

    # --------------------------------------------------------------
    # LLM Analysis
    # --------------------------------------------------------------

    user_msg = f"""
    Source:
    {source}

    Destination:
    {destination}

    Travelers:
    {travelers}

    Trip Duration:
    {days} days

    Budget:
    {budget} {currency}

    Flight Data:
    {flight_data}

    Provide:

    1. Best airline options (2-3 choices)
    2. Cost analysis
    3. Best booking period
    4. Layover recommendations
    5. Baggage tips
    6. Comfort vs value recommendation

    Keep response under 300 words.
    """

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "flight_data": flight_data,
        "flight_summary": response.content,
    }