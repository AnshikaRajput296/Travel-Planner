"""
tools/currency_tool.py
-----------------------
Mock currency conversion tool (INR base).
"""

from langchain_core.tools import tool


# Approximate INR exchange rates (as of 2024)
RATES_FROM_INR = {
    "USD": 0.012,  "EUR": 0.011,  "GBP": 0.0094,
    "JPY": 1.78,   "AED": 0.044,  "SGD": 0.016,
    "THB": 0.43,   "IDR": 187.0,  "MYR": 0.056,
    "AUD": 0.018,  "NZD": 0.020,  "CAD": 0.016,
    "CHF": 0.011,  "ZAR": 0.22,   "NPR": 1.60,
    "LKR": 3.73,   "BDT": 1.32,   "PKR": 3.35,
    "MXN": 0.20,   "BRL": 0.062,  "EGP": 0.58,
    "TRY": 0.39,   "INR": 1.0,
}


@tool
def convert_currency(amount_inr: float, target_currency: str) -> dict:
    """
    Convert INR amount to target currency.

    Args:
        amount_inr:       Amount in Indian Rupees.
        target_currency:  3-letter ISO currency code (e.g. "USD", "JPY").

    Returns:
        Dictionary with converted amount and rate.
    """
    code = target_currency.upper().strip()
    rate = RATES_FROM_INR.get(code, 0.012)
    converted = round(amount_inr * rate, 2)
    return {
        "from": f"₹{amount_inr:,.0f} INR",
        "to": f"{converted:,.2f} {code}",
        "rate": f"1 INR = {rate} {code}",
        "note": "Rates are approximate. Check live rates before travel.",
    }
