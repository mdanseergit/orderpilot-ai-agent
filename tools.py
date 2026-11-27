"""
tools.py - Tool Functions, Gemini Schemas, and Explicit Tool Registry

Defines the core `lookup_order_status` tool and registers it with the
official Google Gen AI SDK schema.
"""

from typing import Any, Dict
from google.genai import types

from database import ORDERS


# ---------------------------------------------------------------------------
# 1. Core Tool Function
# ---------------------------------------------------------------------------
def lookup_order_status(order_id: str) -> Dict[str, Any]:
    """Look up shipping status, delivery estimate, customer, and product details.

    Args:
        order_id: The unique identifier for the order (e.g. 'ORD-101', 'A123').

    Returns:
        A dictionary containing:
        - found (bool): True if order exists, False otherwise.
        - order_id (str): Normalized order ID.
        - customer_name (str, optional): Name of the customer.
        - product (str, optional): Product name.
        - status (str, optional): Order status (Processing, Shipped, etc.).
        - delivery_date (str or None, optional): Estimated delivery date.
        - message (str, optional): Explanation if not found or invalid.
    """
    if not isinstance(order_id, str):
        return {
            "found": False,
            "order_id": str(order_id) if order_id is not None else "",
            "message": "Invalid order ID format. Expected a string.",
        }

    clean_id = order_id.strip().upper()

    if not clean_id:
        return {
            "found": False,
            "order_id": order_id,
            "message": "Order ID cannot be empty.",
        }

    order = ORDERS.get(clean_id)

    if order is None:
        return {
            "found": False,
            "order_id": clean_id,
            "message": f"Order '{clean_id}' was not found in our database.",
        }

    return {
        "found": True,
        "order_id": order["order_id"],
        "customer_name": order["customer_name"],
        "product": order["product"],
        "status": order["status"],
        "delivery_date": order["delivery_date"],
    }


# ---------------------------------------------------------------------------
# 2. Gemini Tool Schema Declaration
# ---------------------------------------------------------------------------
lookup_order_status_declaration = types.FunctionDeclaration(
    name="lookup_order_status",
    description="Look up the current shipping status, product name, and estimated delivery date of an order using its order ID.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The unique order ID to query (e.g. 'ORD-101', 'A123').",
            }
        },
        "required": ["order_id"],
    },
)

# ---------------------------------------------------------------------------
# 3. Explicit Tool Registry
# ---------------------------------------------------------------------------
# Prevents arbitrary code execution by strictly executing registered functions.
TOOL_REGISTRY: Dict[str, Any] = {
    "lookup_order_status": lookup_order_status,
}

# The Tool bundle to supply to the Gemini API
ORDER_TOOL = types.Tool(
    function_declarations=[lookup_order_status_declaration]
)
