"""
database.py - Mock Order Database for OrderPilot & TicketPilot

Provides simulated orders with diverse statuses, customer details,
products, human-readable delivery strings, and ISO dates for
deterministic refund eligibility evaluation.
"""

from typing import Any, Dict

# Deterministic reference date for evaluating the 30-day refund window
SIMULATED_CURRENT_DATE = "2026-11-28"

# Mock orders database: maps normalized uppercase Order IDs to structured details
ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-101": {
        "order_id": "ORD-101",
        "customer_name": "Alice Johnson",
        "product": "Wireless Noise-Canceling Headphones",
        "status": "Shipped",
        "delivery_date": "Tomorrow by 2:00 PM",
        "delivery_iso_date": "2026-11-29",
    },
    "ORD-102": {
        "order_id": "ORD-102",
        "customer_name": "Bob Smith",
        "product": "Mechanical Gaming Keyboard",
        "status": "Processing",
        "delivery_date": "December 2, 2026",
        "delivery_iso_date": "2026-12-02",
    },
    "ORD-103": {
        "order_id": "ORD-103",
        "customer_name": "Catherine Lee",
        "product": "Ultra-Wide Gaming Monitor 34\"",
        "status": "Delivered",
        "delivery_date": "November 25, 2026",
        "delivery_iso_date": "2026-11-25",
    },
    "ORD-104": {
        "order_id": "ORD-104",
        "customer_name": "David Miller",
        "product": "Ergonomic Office Chair",
        "status": "Out for delivery",
        "delivery_date": "Today by 6:00 PM",
        "delivery_iso_date": "2026-11-28",
    },
    "ORD-105": {
        "order_id": "ORD-105",
        "customer_name": "Emma Watson",
        "product": "USB-C Multi-Port Hub",
        "status": "Cancelled",
        "delivery_date": None,
        "delivery_iso_date": None,
    },
    "ORD-106": {
        "order_id": "ORD-106",
        "customer_name": "Frank Harris",
        "product": "4K Ultra HD Webcam",
        "status": "Processing",
        "delivery_date": "December 4, 2026",
        "delivery_iso_date": "2026-12-04",
    },
    "ORD-107": {
        "order_id": "ORD-107",
        "customer_name": "Grace Kim",
        "product": "Standing Desk Converter",
        "status": "Delivered",
        "delivery_date": "October 10, 2026",
        "delivery_iso_date": "2026-10-10",
    },
    # Backward compatibility with Week 1 starter test IDs (A123 - A128)
    "A123": {
        "order_id": "A123",
        "customer_name": "John Doe",
        "product": "Smart Fitness Watch",
        "status": "Shipped",
        "delivery_date": "Tomorrow",
        "delivery_iso_date": "2026-11-29",
    },
    "A124": {
        "order_id": "A124",
        "customer_name": "Jane Smith",
        "product": "Bluetooth Portable Speaker",
        "status": "Processing",
        "delivery_date": "December 2, 2026",
        "delivery_iso_date": "2026-12-02",
    },
    "A125": {
        "order_id": "A125",
        "customer_name": "Michael Brown",
        "product": "Ergonomic Wireless Mouse",
        "status": "Delivered",
        "delivery_date": "November 25, 2026",
        "delivery_iso_date": "2026-11-25",
    },
    "A126": {
        "order_id": "A126",
        "customer_name": "Sarah Davis",
        "product": "Adjustable Aluminum Laptop Stand",
        "status": "Out for delivery",
        "delivery_date": "Today",
        "delivery_iso_date": "2026-11-28",
    },
    "A127": {
        "order_id": "A127",
        "customer_name": "Robert Wilson",
        "product": "True Wireless Earbuds",
        "status": "Cancelled",
        "delivery_date": None,
        "delivery_iso_date": None,
    },
    "A128": {
        "order_id": "A128",
        "customer_name": "Emily Clark",
        "product": "Leather Desk Pad Mat",
        "status": "Processing",
        "delivery_date": "December 4, 2026",
        "delivery_iso_date": "2026-12-04",
    },
}
