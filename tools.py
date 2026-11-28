"""
tools.py - Tool Functions, Gemini Schemas, and Explicit Tool Registry

Provides four Python tools for TicketPilot (and Week 1 OrderPilot):
1. lookup_order_status(order_id)        - Retrieve order details from database.py.
2. classify_ticket_category(text)       - Deterministic keyword classification.
3. check_refund_eligibility(order_id)   - Deterministic 30-day & status policy check.
4. escalate_to_human(reason, ticket_id) - Optional stretch goal for simulated escalation.
"""

from datetime import date
import re
from typing import Any, Dict, List, Optional
from google.genai import types

from database import ORDERS, SIMULATED_CURRENT_DATE


# ---------------------------------------------------------------------------
# Keyword Rules & Precedence for Ticket Classification
# ---------------------------------------------------------------------------
# Tie-breaking precedence when distinct categories match equal numbers of keywords:
# billing > shipping > bug > other
CATEGORY_PRECEDENCE: List[str] = ["billing", "shipping", "bug"]

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "billing": [
        "payment",
        "charged",
        "charge",
        "invoice",
        "refund",
        "transaction",
        "billing",
        "duplicate",
        "overcharged",
    ],
    "shipping": [
        "missing package",
        "delivery",
        "delivered",
        "tracking",
        "shipment",
        "shipped",
        "courier",
        "package",
        "transit",
        "delay",
        "delayed",
        "arrived",
        "shipping",
    ],
    "bug": [
        "login failure",
        "crash",
        "crashes",
        "crashing",
        "error",
        "broken",
        "malfunction",
        "malfunctioned",
        "login",
        "log in",
        "bug",
        "freeze",
        "freezes",
        "glitch",
        "defect",
    ],
}

# Simulated in-memory escalation queue (never sends live notifications)
ESCALATION_LOG: List[Dict[str, Any]] = []


def reset_escalation_log() -> None:
    """Clear the simulated escalation log (used between runs/tests)."""
    ESCALATION_LOG.clear()


def get_escalation_count() -> int:
    """Return total number of simulated escalations recorded."""
    return len(ESCALATION_LOG)


# ---------------------------------------------------------------------------
# TOOL 1: Order Status Lookup (Preserved from Week 1)
# ---------------------------------------------------------------------------
def lookup_order_status(order_id: str) -> Dict[str, Any]:
    """Look up shipping status, delivery estimate, customer, and product details.

    Args:
        order_id: The unique identifier for the order (e.g. 'ORD-101', 'A123').

    Returns:
        Structured dict with order details or a graceful not-found message.
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
        "delivery_iso_date": order.get("delivery_iso_date"),
    }


# ---------------------------------------------------------------------------
# TOOL 2: Keyword-Based Ticket Classification
# ---------------------------------------------------------------------------
def classify_ticket_category(text: str) -> Dict[str, Any]:
    """Classify support ticket text into 'billing', 'shipping', 'bug', or 'other'.

    Scoring & Precedence:
    - Counts distinct keyword matches per category using word boundaries.
    - Highest match count wins.
    - Ties are resolved deterministically using precedence:
      billing > shipping > bug > other.

    Args:
        text: Combined subject and body text of the support ticket.

    Returns:
        Dict with 'category', 'reason', and 'matched_keywords'.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "category": "other",
            "reason": "Input text was empty or invalid; defaulted to 'other'.",
            "matched_keywords": {},
        }

    text_lower = text.lower()
    matched_by_category: Dict[str, List[str]] = {}

    for category in CATEGORY_PRECEDENCE:
        matches: List[str] = []
        for keyword in CATEGORY_KEYWORDS[category]:
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                matches.append(keyword)
        if matches:
            matched_by_category[category] = matches

    if not matched_by_category:
        return {
            "category": "other",
            "reason": "No billing, shipping, or bug keywords matched; classified as 'other'.",
            "matched_keywords": {},
        }

    # Pick category with highest keyword count; tie-break by CATEGORY_PRECEDENCE order
    best_category = max(
        CATEGORY_PRECEDENCE,
        key=lambda cat: (
            len(matched_by_category.get(cat, [])),
            -CATEGORY_PRECEDENCE.index(cat),
        ),
    )

    winning_terms = matched_by_category[best_category]
    tie_note = ""
    top_count = len(winning_terms)
    tied = [c for c, terms in matched_by_category.items() if len(terms) == top_count]
    if len(tied) > 1:
        tie_note = (
            f" Tie detected among {tied} (each matched {top_count} term(s)); "
            f"resolved via precedence (billing > shipping > bug)."
        )

    return {
        "category": best_category,
        "reason": f"Matched {best_category} keyword(s): {', '.join(winning_terms)}.{tie_note}",
        "matched_keywords": matched_by_category,
    }


# ---------------------------------------------------------------------------
# TOOL 3: Deterministic Refund Eligibility Policy Check
# ---------------------------------------------------------------------------
def check_refund_eligibility(
    order_id: str,
    current_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Check whether an order meets the fictional refund policy without issuing a refund.

    Deterministic Policy:
    - Delivered orders: eligible if delivered within 30 days of current_date.
    - Cancelled orders: eligible for automatic refund approval.
    - Processing / Shipped / Out for delivery: not automatically eligible;
      requires human review before refund approval.
    - Unknown / invalid order ID: not eligible for automatic approval.

    Args:
        order_id: Order ID to evaluate.
        current_date: Optional ISO 'YYYY-MM-DD' override (defaults to SIMULATED_CURRENT_DATE).

    Returns:
        Structured dictionary with 'eligible' (bool), 'reason' (str),
        'requires_human_review' (bool), and 'refund_issued' (always False).
    """
    if not isinstance(order_id, str) or not order_id.strip():
        return {
            "order_id": str(order_id) if order_id is not None else "",
            "found": False,
            "eligible": False,
            "requires_human_review": False,
            "refund_issued": False,
            "policy_rule": "invalid_order_id",
            "reason": "Missing or invalid order ID. Cannot verify refund eligibility.",
        }

    clean_id = order_id.strip().upper()
    order = ORDERS.get(clean_id)

    if order is None:
        return {
            "order_id": clean_id,
            "found": False,
            "eligible": False,
            "requires_human_review": True,
            "refund_issued": False,
            "policy_rule": "unknown_order",
            "reason": f"Order '{clean_id}' does not exist in our records and is not eligible for automatic refund.",
        }

    status = order["status"]

    if status == "Cancelled":
        return {
            "order_id": clean_id,
            "found": True,
            "status": status,
            "eligible": True,
            "requires_human_review": False,
            "refund_issued": False,
            "policy_rule": "cancelled_order",
            "reason": f"Order '{clean_id}' was cancelled and is eligible for a full refund.",
        }

    if status == "Delivered":
        iso_str = order.get("delivery_iso_date")
        if not iso_str:
            return {
                "order_id": clean_id,
                "found": True,
                "status": status,
                "eligible": False,
                "requires_human_review": True,
                "refund_issued": False,
                "policy_rule": "missing_delivery_date",
                "reason": f"Order '{clean_id}' is marked Delivered but lacks a verified delivery date; human review required.",
            }

        ref_date = date.fromisoformat(current_date or SIMULATED_CURRENT_DATE)
        delivered_date = date.fromisoformat(iso_str)
        days_elapsed = (ref_date - delivered_date).days

        if 0 <= days_elapsed <= 30:
            return {
                "order_id": clean_id,
                "found": True,
                "status": status,
                "days_since_delivery": days_elapsed,
                "eligible": True,
                "requires_human_review": False,
                "refund_issued": False,
                "policy_rule": "delivered_within_30_days",
                "reason": (
                    f"Order '{clean_id}' was delivered {days_elapsed} day(s) ago (on {iso_str}), "
                    "which is within the 30-day refund window."
                ),
            }

        return {
            "order_id": clean_id,
            "found": True,
            "status": status,
            "days_since_delivery": days_elapsed,
            "eligible": False,
            "requires_human_review": False,
            "refund_issued": False,
            "policy_rule": "delivered_outside_30_days",
            "reason": (
                f"Order '{clean_id}' was delivered {days_elapsed} day(s) ago (on {iso_str}), "
                "which exceeds the 30-day refund eligibility window."
            ),
        }

    # Processing, Shipped, Out for delivery
    return {
        "order_id": clean_id,
        "found": True,
        "status": status,
        "eligible": False,
        "requires_human_review": True,
        "refund_issued": False,
        "policy_rule": "in_progress_requires_review",
        "reason": (
            f"Order '{clean_id}' is currently '{status}'. "
            "Processing, shipped, and out-for-delivery orders require human review before refund approval."
        ),
    }


# ---------------------------------------------------------------------------
# TOOL 4 (Optional Stretch Goal): Simulated Human Escalation
# ---------------------------------------------------------------------------
def escalate_to_human(reason: str, ticket_id: Optional[str] = None) -> Dict[str, Any]:
    """Record a simulated escalation for cases requiring human review.

    This tool NEVER sends real notifications or contacts an actual human.

    Args:
        reason: Concise explanation of why human intervention is needed.
        ticket_id: Optional ticket identifier being escalated.

    Returns:
        Structured confirmation dict containing simulated escalation metadata.
    """
    if not isinstance(reason, str) or not reason.strip():
        return {
            "escalated": False,
            "simulated": True,
            "error": "Escalation reason cannot be empty.",
        }

    clean_reason = reason.strip()
    escalation_id = f"ESC-{len(ESCALATION_LOG) + 1:03d}"
    record = {
        "escalation_id": escalation_id,
        "ticket_id": ticket_id,
        "reason": clean_reason,
        "status": "queued_for_review",
        "simulated": True,
    }
    ESCALATION_LOG.append(record)

    return {
        "escalated": True,
        "escalation_id": escalation_id,
        "ticket_id": ticket_id,
        "status": "queued_for_review",
        "reason": clean_reason,
        "simulated": True,
        "note": "Simulated escalation recorded in local triage queue. No live human notification was sent.",
    }


# ---------------------------------------------------------------------------
# Gemini Tool Schema Declarations
# ---------------------------------------------------------------------------
lookup_order_status_declaration = types.FunctionDeclaration(
    name="lookup_order_status",
    description=(
        "Look up the current shipping status, product name, customer name, and estimated "
        "delivery date of an order using its order ID."
    ),
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

classify_ticket_category_declaration = types.FunctionDeclaration(
    name="classify_ticket_category",
    description=(
        "Classify a customer support ticket into one of four categories ('billing', 'shipping', "
        "'bug', or 'other') based on its subject and message text."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The combined subject and body of the support ticket to classify.",
            }
        },
        "required": ["text"],
    },
)

check_refund_eligibility_declaration = types.FunctionDeclaration(
    name="check_refund_eligibility",
    description=(
        "Evaluate whether an order is eligible for a product or order refund under store policy "
        "(30-day delivery window, cancelled orders, or requiring human review). Call ONLY when the "
        "customer explicitly asks for a refund, return, or refund eligibility—do NOT call for "
        "general invoice discrepancies or duplicate payment investigations. Never issues actual refunds."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The unique order ID to evaluate for refund eligibility.",
            }
        },
        "required": ["order_id"],
    },
)

escalate_to_human_declaration = types.FunctionDeclaration(
    name="escalate_to_human",
    description=(
        "Record a simulated escalation to a human review queue for urgent, fraudulent, "
        "disputed, or policy-blocked tickets requiring manual review."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Clear summary of why human review is required.",
            },
            "ticket_id": {
                "type": "string",
                "description": "Optional support ticket ID (e.g. 'T012').",
            },
        },
        "required": ["reason"],
    },
)

# Explicit registry preventing arbitrary code execution
TOOL_REGISTRY: Dict[str, Any] = {
    "lookup_order_status": lookup_order_status,
    "classify_ticket_category": classify_ticket_category,
    "check_refund_eligibility": check_refund_eligibility,
    "escalate_to_human": escalate_to_human,
}

# Preserved Week 1 single-tool bundle
ORDER_TOOL = types.Tool(
    function_declarations=[lookup_order_status_declaration]
)

# Week 2 multi-tool bundle for TicketPilot
TICKET_TOOLS = types.Tool(
    function_declarations=[
        lookup_order_status_declaration,
        classify_ticket_category_declaration,
        check_refund_eligibility_declaration,
        escalate_to_human_declaration,
    ]
)
