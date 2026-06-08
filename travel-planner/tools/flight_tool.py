"""
tools/flight_tool.py
---------------------
Mock flight search tool used by the Flight Agent.
Returns realistic flight options based on origin/destination.
"""

from typing import Any
from langchain_core.tools import tool


@tool
def search_flights(destination: str, budget_inr: float, travelers: int) -> dict[str, Any]:
    """
    Search for available flights to a destination.

    Args:
        destination: Target city or country.
        budget_inr:  Total trip budget in INR.
        travelers:   Number of passengers.

    Returns:
        Dictionary with flight options and cost estimates.
    """
    # Mock realistic per-person flight cost heuristics (one-way in INR)
    cost_map = {
        "japan": 45_000, "tokyo": 45_000,
        "usa": 55_000, "new york": 60_000, "los angeles": 58_000,
        "europe": 40_000, "paris": 42_000, "london": 40_000, "rome": 38_000,
        "dubai": 18_000, "uae": 18_000,
        "thailand": 12_000, "bangkok": 12_000,
        "singapore": 15_000,
        "bali": 14_000, "indonesia": 14_000,
        "maldives": 20_000,
        "australia": 55_000, "sydney": 55_000,
        "sri lanka": 8_000,
        "nepal": 6_000, "kathmandu": 6_000,
        "bhutan": 10_000,
        "malaysia": 13_000, "kuala lumpur": 13_000,
        "vietnam": 11_000, "hanoi": 11_000,
        "cambodia": 13_000,
        "egypt": 30_000, "cairo": 30_000,
        "kenya": 35_000,
        "south africa": 40_000, "cape town": 40_000,
        "canada": 58_000,
        "brazil": 55_000, "rio": 55_000,
        "argentina": 52_000,
        "mexico": 50_000,
        "turkey": 28_000, "istanbul": 28_000,
        "greece": 35_000, "athens": 35_000,
        "spain": 38_000, "barcelona": 38_000,
        "switzerland": 42_000, "zurich": 42_000,
        "germany": 38_000, "berlin": 38_000,
        "goa": 4_000, "delhi": 3_500, "mumbai": 3_500,
        "kerala": 5_000, "rajasthan": 4_500,
    }

    dest_lower = destination.lower()
    base_cost = 20_000  # default
    for key, val in cost_map.items():
        if key in dest_lower:
            base_cost = val
            break

    round_trip_per_person = base_cost * 2
    total_flight_cost = round_trip_per_person * travelers

    airlines = {
        "japan": ["Air India", "Japan Airlines", "ANA"],
        "usa": ["Air India", "United Airlines", "Delta"],
        "europe": ["Air India", "Lufthansa", "British Airways"],
        "dubai": ["IndiGo", "Air Arabia", "Emirates"],
        "thailand": ["IndiGo", "Thai Airways", "AirAsia"],
        "singapore": ["IndiGo", "Singapore Airlines", "Scoot"],
        "bali": ["IndiGo", "Batik Air", "Lion Air"],
        "maldives": ["Air India", "Maldivian", "IndiGo"],
    }

    airline_list = ["Air India", "IndiGo", "SpiceJet"]
    for key, vals in airlines.items():
        if key in dest_lower:
            airline_list = vals
            break

    return {
        "destination": destination,
        "airlines": airline_list,
        "cost_per_person_inr": round_trip_per_person,
        "total_cost_inr": total_flight_cost,
        "duration_hours": _estimate_flight_hours(dest_lower),
        "recommended_booking": "Book 6–8 weeks in advance for best prices",
        "best_travel_months": _best_months(dest_lower),
    }


def _estimate_flight_hours(dest_lower: str) -> str:
    mapping = {
        "japan": "7–9", "usa": "14–17", "europe": "9–11",
        "dubai": "3–4", "thailand": "3–5", "singapore": "5–6",
        "bali": "5–7", "maldives": "4–5", "australia": "11–13",
        "goa": "1–2", "delhi": "0.5–1", "mumbai": "0.5–1",
    }
    for key, val in mapping.items():
        if key in dest_lower:
            return val
    return "8–12"


def _best_months(dest_lower: str) -> str:
    mapping = {
        "japan": "March–April (Cherry Blossom) / Oct–Nov",
        "thailand": "November–February",
        "bali": "April–October",
        "europe": "May–September",
        "maldives": "November–April",
        "dubai": "November–March",
    }
    for key, val in mapping.items():
        if key in dest_lower:
            return val
    return "Year-round (check local seasons)"
