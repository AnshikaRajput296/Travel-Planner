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


SYSTEM_PROMPT = """You are a luxury travel consultant and accommodation expert.
You suggest the best hotels, hostels, and resorts for every budget tier.
Be specific about hotel names, locations, amenities, and value propositions.
Always provide Budget, Mid-range, and Luxury options with honest pros and cons."""


def run_hotel_agent(state: dict[str, Any], llm: ChatGroq) -> dict[str, Any]:
    """
    Execute the Hotel Agent node.

    Args:
        state: Shared LangGraph state.
        llm:   ChatGroq instance.

    Returns:
        Updated state with 'hotel_data' and 'hotel_summary'.
    """
    destination  = state["destination"]
    budget       = state["budget"]
    days         = state["days"]
    travelers    = state["travelers"]
    preferences  = state.get("preferences", [])
    flight_cost  = state.get("flight_data", {}).get("total_cost_inr", 0)

    remaining_after_flights = budget - flight_cost
    budget_per_night        = remaining_after_flights / max(days, 1) * 0.35  # ~35 % for accommodation

    hotel_info = search_hotels.invoke({
        "destination": destination,
        "budget_per_night_inr": budget_per_night,
        "days": days,
        "travelers": travelers,
    })

    user_msg = (
        f"Destination: {destination}\n"
        f"Trip: {days} days, {travelers} traveler(s)\n"
        f"Preferences: {', '.join(preferences) if preferences else 'General'}\n"
        f"Hotel budget (approx): ₹{budget_per_night:,.0f}/night\n"
        f"Tool data: {hotel_info}\n\n"
        "Provide:\n"
        "1. Budget option (₹/night + location highlights)\n"
        "2. Mid-range option (best value, ₹/night)\n"
        "3. Luxury option (₹/night + USP)\n"
        "4. Booking tips (best platforms, cancel policy)\n"
        "5. Neighborhood recommendation\n"
        "Keep under 300 words. Use ₹ for prices."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "hotel_data": hotel_info,
        "hotel_summary": response.content,
    }
