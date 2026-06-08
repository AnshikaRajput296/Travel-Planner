"""
tools/hotel_tool.py
--------------------
Mock hotel search tool used by the Hotel Agent.
"""

from typing import Any
from langchain_core.tools import tool


@tool
def search_hotels(
    destination: str,
    budget_per_night_inr: float,
    days: int,
    travelers: int,
) -> dict[str, Any]:
    """
    Search for hotel options at the destination.

    Args:
        destination:         City/country name.
        budget_per_night_inr: Target nightly budget in INR.
        days:                 Number of nights.
        travelers:            Number of guests.

    Returns:
        Dictionary with budget / mid-range / luxury hotel suggestions.
    """
    dest_lower = destination.lower()

    # Cost multiplier relative to India baseline
    multiplier_map = {
        "japan": 4.5, "tokyo": 4.5,
        "usa": 5.5, "new york": 7.0, "los angeles": 6.0,
        "europe": 5.0, "paris": 6.5, "london": 7.0, "rome": 5.5,
        "dubai": 4.0, "singapore": 5.0,
        "thailand": 2.0, "bangkok": 2.0,
        "bali": 2.5, "maldives": 8.0,
        "australia": 6.0, "sydney": 6.5,
        "sri lanka": 1.5, "nepal": 1.2,
        "malaysia": 2.0, "vietnam": 1.5,
        "egypt": 2.5, "turkey": 3.0,
        "greece": 4.0, "spain": 4.5,
        "switzerland": 7.5, "germany": 5.0,
        "goa": 1.5, "kerala": 1.3, "rajasthan": 1.4,
    }

    mult = 3.0
    for key, val in multiplier_map.items():
        if key in dest_lower:
            mult = val
            break

    base = 1_500  # INR baseline per room night

    budget_hotel    = {"name": f"Budget Inn {destination}", "price": round(base * mult * 0.6), "rating": 3.2, "type": "Budget"}
    midrange_hotel  = {"name": f"Comfort Suites {destination}", "price": round(base * mult * 1.0), "rating": 4.0, "type": "Mid-range"}
    luxury_hotel    = {"name": f"Grand Palace {destination}", "price": round(base * mult * 2.5), "rating": 4.7, "type": "Luxury"}

    rooms_needed = max(1, (travelers + 1) // 2)

    return {
        "destination": destination,
        "nights": days,
        "rooms_needed": rooms_needed,
        "options": [budget_hotel, midrange_hotel, luxury_hotel],
        "recommended": midrange_hotel if budget_per_night_inr >= midrange_hotel["price"] else budget_hotel,
        "total_hotel_cost": round(
            midrange_hotel["price"] * days * rooms_needed
            if budget_per_night_inr >= midrange_hotel["price"]
            else budget_hotel["price"] * days * rooms_needed
        ),
        "booking_tips": "Book via Booking.com or Agoda for best deals. Cancel-free options recommended.",
    }
