"""
graph/travel_graph.py
----------------------
LangGraph StateGraph that orchestrates all travel agents.

Workflow:

START
  ↓
flight_agent
  ↓
hotel_agent
  ↓
budget_agent
  ↓
itinerary_agent
  ↓
recommendation_agent
  ↓
END
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from agents.flight_agent import run_flight_agent
from agents.hotel_agent import run_hotel_agent
from agents.budget_agent import run_budget_agent
from agents.itinerary_agent import run_itinerary_agent
from agents.recommendation_agent import run_recommendation_agent

load_dotenv()


# ------------------------------------------------------------------
# Shared State Schema
# ------------------------------------------------------------------

class TravelState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.
    """

    # --------------------------------------------------------------
    # User Inputs
    # --------------------------------------------------------------

    source: str
    destination: str

    budget: float
    days: int
    travelers: int

    preferences: list[str]
    travel_dates: str

    # --------------------------------------------------------------
    # Flight Agent
    # --------------------------------------------------------------

    flight_data: dict
    flight_summary: str

    # --------------------------------------------------------------
    # Hotel Agent
    # --------------------------------------------------------------

    hotel_data: dict
    hotel_summary: str

    # --------------------------------------------------------------
    # Budget Agent
    # --------------------------------------------------------------

    budget_breakdown: dict
    budget_summary: str

    # --------------------------------------------------------------
    # Itinerary Agent
    # --------------------------------------------------------------

    itinerary: str

    # --------------------------------------------------------------
    # Recommendation Agent
    # --------------------------------------------------------------

    recommendations: str

    # --------------------------------------------------------------
    # Global Metadata
    # --------------------------------------------------------------

    currency: str

    status: str
    error: str | None


# ------------------------------------------------------------------
# Graph Builder
# ------------------------------------------------------------------

def build_travel_graph():
    """
    Build and compile LangGraph workflow.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.4,
        max_tokens=2048,
    )

    graph = StateGraph(TravelState)

    # --------------------------------------------------------------
    # Node Wrappers
    # --------------------------------------------------------------

    def flight_node(state: TravelState) -> TravelState:
        return run_flight_agent(state, llm)

    def hotel_node(state: TravelState) -> TravelState:
        return run_hotel_agent(state, llm)

    def budget_node(state: TravelState) -> TravelState:
        return run_budget_agent(state, llm)

    def itinerary_node(state: TravelState) -> TravelState:
        return run_itinerary_agent(state, llm)

    def recommendation_node(state: TravelState) -> TravelState:
        return run_recommendation_agent(state, llm)

    # --------------------------------------------------------------
    # Register Nodes
    # --------------------------------------------------------------

    graph.add_node("flight_agent", flight_node)
    graph.add_node("hotel_agent", hotel_node)
    graph.add_node("budget_agent", budget_node)
    graph.add_node("itinerary_agent", itinerary_node)
    graph.add_node("recommendation_agent", recommendation_node)

    # --------------------------------------------------------------
    # Workflow Edges
    # --------------------------------------------------------------

    graph.add_edge(START, "flight_agent")

    graph.add_edge(
        "flight_agent",
        "hotel_agent",
    )

    graph.add_edge(
        "hotel_agent",
        "budget_agent",
    )

    graph.add_edge(
        "budget_agent",
        "itinerary_agent",
    )

    graph.add_edge(
        "itinerary_agent",
        "recommendation_agent",
    )

    graph.add_edge(
        "recommendation_agent",
        END,
    )

    return graph.compile()


# ------------------------------------------------------------------
# Pipeline Runner
# ------------------------------------------------------------------

def run_travel_pipeline(
    source: str,
    destination: str,
    budget: float,
    days: int,
    travelers: int,
    preferences: list[str],
    travel_dates: str = "",
) -> dict[str, Any]:
    """
    Execute full travel planning workflow.
    """

    app = build_travel_graph()

    initial_state: TravelState = {
        "source": source,
        "destination": destination,
        "budget": budget,
        "days": days,
        "travelers": travelers,
        "preferences": preferences,
        "travel_dates": travel_dates,
        "status": "running",
        "error": None,
    }

    try:
        result = app.invoke(initial_state)

        result["status"] = "completed"

        return result

    except Exception as e:

        return {
            **initial_state,
            "status": "failed",
            "error": str(e),
        }