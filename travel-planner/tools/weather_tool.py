"""
tools/weather_tool.py
----------------------
Mock weather information tool.
"""

from langchain_core.tools import tool


@tool
def get_weather(destination: str, travel_month: str = "any") -> dict:
    """
    Get weather information for a destination.

    Args:
        destination:  City or country.
        travel_month: Month of travel (optional).

    Returns:
        Weather info dict.
    """
    dest_lower = destination.lower()

    profiles = {
        "japan":     {"avg_temp": "10–25°C", "condition": "Mild, seasonal", "best": "Spring/Autumn", "rain": "June–July (rainy season)"},
        "thailand":  {"avg_temp": "25–35°C", "condition": "Tropical", "best": "Nov–Feb", "rain": "May–October"},
        "bali":      {"avg_temp": "26–32°C", "condition": "Tropical humid", "best": "April–Oct", "rain": "Nov–March"},
        "dubai":     {"avg_temp": "20–42°C", "condition": "Desert hot", "best": "Nov–March", "rain": "Rare"},
        "europe":    {"avg_temp": "5–25°C", "condition": "Temperate", "best": "May–Sept", "rain": "Year-round"},
        "maldives":  {"avg_temp": "26–30°C", "condition": "Tropical", "best": "Nov–April", "rain": "May–October"},
        "usa":       {"avg_temp": "10–30°C", "condition": "Varies by region", "best": "May–Sept", "rain": "Spring/Summer"},
        "australia": {"avg_temp": "15–35°C", "condition": "Varied", "best": "Sept–Nov / March–May", "rain": "Varies"},
        "singapore": {"avg_temp": "25–32°C", "condition": "Equatorial", "best": "Feb–April", "rain": "Nov–Jan"},
        "nepal":     {"avg_temp": "5–25°C", "condition": "Mountain climate", "best": "Oct–Nov / March–May", "rain": "June–Sept (monsoon)"},
        "india":     {"avg_temp": "20–40°C", "condition": "Tropical/Semi-arid", "best": "Oct–March", "rain": "June–Sept (monsoon)"},
        "goa":       {"avg_temp": "24–33°C", "condition": "Tropical coastal", "best": "Nov–Feb", "rain": "June–Sept"},
        "kerala":    {"avg_temp": "22–32°C", "condition": "Tropical", "best": "Sept–March", "rain": "June–August"},
    }

    weather = {"avg_temp": "20–30°C", "condition": "Moderate", "best": "Year-round", "rain": "Occasional"}
    for key, val in profiles.items():
        if key in dest_lower:
            weather = val
            break

    weather["destination"] = destination
    weather["travel_tip"] = f"Best time to visit: {weather['best']}. Pack accordingly for {weather['condition']} conditions."
    return weather
