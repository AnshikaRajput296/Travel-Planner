"""
agents/itinerary_agent.py
--------------------------
Itinerary Agent: generates a detailed day-wise travel plan.
"""

from __future__ import annotations
from typing import Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are an expert travel itinerary designer with deep knowledge of global destinations.
You craft immersive, practical, day-by-day travel plans that blend iconic sights with hidden gems.
Always include: morning / afternoon / evening activities, meal suggestions, and transport between spots.
Format each day clearly as "Day N: [Theme]" with bullet points for activities."""


def run_itinerary_agent(state: dict[str, Any], llm: ChatGroq) -> dict[str, Any]:
    """
    Execute the Itinerary Agent node.

    Generates a detailed day-wise plan based on destination, duration,
    and user preferences.
    """
    destination = state["destination"]
    days        = state["days"]
    travelers   = state["travelers"]
    preferences = state.get("preferences", [])
    budget      = state["budget"]

    pref_str = ", ".join(preferences) if preferences else "General sightseeing"

    user_msg = (
        f"Create a detailed {days}-day itinerary for {destination}.\n"
        f"Travelers: {travelers} | Preferences: {pref_str}\n"
        f"Total Budget: ₹{budget:,.0f}\n\n"
        "For EACH day include:\n"
        "• A thematic title (e.g. 'Day 1: City Landmarks')\n"
        "• Morning activity (8:00–12:00)\n"
        "• Lunch recommendation with cuisine type\n"
        "• Afternoon activity (13:00–17:00)\n"
        "• Evening activity / dinner suggestion (18:00–21:00)\n"
        "• Estimated daily spend in ₹\n"
        "• One local tip or insider secret\n\n"
        f"Keep each day to ~150 words. Total response under {days * 160 + 100} words."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "itinerary": response.content,
    }
