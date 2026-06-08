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


SYSTEM_PROMPT = """You are an expert flight booking assistant specializing in travel from India.
Your job is to analyze flight options and provide accurate cost estimates and recommendations.
Always respond in a structured, clear format. Focus on value for money and convenience.
Provide specific airline names, approximate costs in INR, and practical travel tips."""


def run_flight_agent(state: dict[str, Any], llm: ChatGroq) -> dict[str, Any]:
    """
    Execute the Flight Agent node.

    Uses the search_flights tool to get mock data, then calls the LLM
    to produce a polished flight recommendation narrative.

    Args:
        state: Shared LangGraph state dict.
        llm:   Configured ChatGroq instance.

    Returns:
        Updated state with 'flight_data' and 'flight_summary' keys.
    """
    destination = state["destination"]
    budget      = state["budget"]
    travelers   = state["travelers"]
    days        = state["days"]

    # ── Tool call ────────────────────────────────────────────────────────────
    flight_info = search_flights.invoke({
        "destination": destination,
        "budget_inr": budget,
        "travelers": travelers,
    })

    # ── LLM enrichment ───────────────────────────────────────────────────────
    user_msg = (
        f"Destination: {destination}\n"
        f"Budget: ₹{budget:,.0f} for {travelers} traveler(s), {days} days\n"
        f"Tool data: {flight_info}\n\n"
        "Based on this data, provide:\n"
        "1. Recommended airlines (2–3 options with pros/cons)\n"
        "2. Estimated round-trip cost per person in INR\n"
        "3. Best time to book and travel\n"
        "4. Flight duration and layover tips\n"
        "5. Baggage and comfort recommendations\n"
        "Keep the response concise (under 300 words). Use ₹ for prices."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)
    summary  = response.content

    return {
        **state,
        "flight_data": flight_info,
        "flight_summary": summary,
    }
