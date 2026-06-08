"""
agents/recommendation_agent.py
--------------------------------
Recommendation Agent: curates restaurants, attractions, hidden gems, and travel tips.
"""

from __future__ import annotations
from typing import Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are a seasoned travel blogger and local insider who has lived in destinations worldwide.
You reveal the best restaurants, hidden gems, local experiences, and practical travel tips.
Your recommendations go beyond tourist traps — you know the local favorites.
Always split into: 🍽️ Restaurants | 🏛️ Attractions | 💎 Hidden Gems | 💡 Travel Tips."""


def run_recommendation_agent(state: dict[str, Any], llm: ChatGroq) -> dict[str, Any]:
    """
    Execute the Recommendation Agent node.

    Produces curated restaurant, attraction, and tip recommendations.
    """
    destination = state["destination"]
    days        = state["days"]
    preferences = state.get("preferences", [])
    travelers   = state["travelers"]

    pref_str = ", ".join(preferences) if preferences else "General"

    user_msg = (
        f"Destination: {destination} | {days} days | Interests: {pref_str}\n"
        f"Travelers: {travelers}\n\n"
        "Provide curated recommendations:\n\n"
        "🍽️ TOP RESTAURANTS (5 picks)\n"
        "  - Name, cuisine, price range (₹/person), must-try dish\n\n"
        "🏛️ TOP ATTRACTIONS (5 picks)\n"
        "  - Name, why it's special, best time to visit, entry fee\n\n"
        "💎 HIDDEN GEMS (3 picks)\n"
        "  - Off-the-beaten-path spots most tourists miss\n\n"
        "💡 INSIDER TRAVEL TIPS (5 tips)\n"
        "  - Practical local knowledge, cultural etiquette, scam warnings\n\n"
        "🛍️ SHOPPING SPOTS (3 picks if relevant)\n"
        "  - Best markets, what to buy, price negotiation tips\n\n"
        "Keep each item brief (1–2 lines). Total under 400 words."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "recommendations": response.content,
    }
