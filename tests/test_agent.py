"""
test_agent.py - Offline Multi-Round Agent Tests (Mocked Gemini API)

Covers:
- Test I: Refund ticket requiring multiple tools across multiple rounds before final answer.
- Test J: Ticket requiring classification only (no order lookup).
- Test L: Malformed tool arguments, unknown tool names, invalid tool responses,
  duplicate prevention, and Week 1 backward-compatibility.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent import (
    TICKET_SYSTEM_INSTRUCTION,
    triage_ticket,
    validate_and_execute_tool,
    validate_final_output,
)
from tools import check_refund_eligibility_declaration
from main import process_user_query


def _make_call(name, args, call_id="call_1"):
    call = MagicMock()
    call.name = name
    call.args = args
    call.id = call_id
    return call


def _make_response(function_calls=None, text=None):
    resp = MagicMock()
    resp.function_calls = function_calls
    resp.text = text
    candidate = MagicMock()
    candidate.content = MagicMock()
    resp.candidates = [candidate]
    return resp


class TestTicketPilotAgent(unittest.TestCase):
    """Offline unit tests for the multi-round TicketPilot agent loop."""

    # -----------------------------------------------------------------------
    # Test I: Refund ticket requiring two+ different tools across multiple rounds
    # -----------------------------------------------------------------------
    def test_i_multi_round_refund_ticket(self):
        mock_client = MagicMock()

        # Round 1: Gemini asks to classify and look up the order
        round1 = _make_response(
            function_calls=[
                _make_call("classify_ticket_category", {"text": "Refund for cancelled hub order ORD-105"}, "c1"),
                _make_call("lookup_order_status", {"order_id": "ORD-105"}, "c2"),
            ]
        )
        # Round 2: Gemini sees order is Cancelled and asks to check refund eligibility
        round2 = _make_response(
            function_calls=[
                _make_call("check_refund_eligibility", {"order_id": "ORD-105"}, "c3"),
            ]
        )
        # Final Round: Gemini outputs valid structured JSON
        final_json = json.dumps(
            {
                "ticket_id": "T003",
                "category": "billing",
                "tools_called": [
                    "classify_ticket_category",
                    "lookup_order_status",
                    "check_refund_eligibility",
                ],
                "draft_reply": (
                    "Hello Emma, your order ORD-105 (USB-C Multi-Port Hub) was cancelled "
                    "and is eligible for a full refund under our policy."
                ),
            }
        )
        round3 = _make_response(function_calls=None, text=final_json)

        mock_client.models.generate_content.side_effect = [round1, round2, round3]

        ticket = {
            "ticket_id": "T003",
            "subject": "Refund for cancelled hub order",
            "body": "My order ORD-105 was cancelled. Am I eligible for a refund?",
            "customer_email": "emma.watson@example.com",
            "order_id": "ORD-105",
        }

        result = triage_ticket(ticket, mock_client, model_name="gemini-3.5-flash", verbose=False)

        self.assertIsNone(result["error"])
        self.assertFalse(result["safety_cap_reached"])
        self.assertEqual(result["ticket_id"], "T003")
        self.assertEqual(result["category"], "billing")
        self.assertEqual(result["tool_rounds"], 2)
        self.assertEqual(
            result["tools_called"],
            ["classify_ticket_category", "lookup_order_status", "check_refund_eligibility"],
        )
        self.assertIn("eligible for a full refund", result["draft_reply"])
        self.assertEqual(mock_client.models.generate_content.call_count, 3)

    # -----------------------------------------------------------------------
    # Test J: Ticket requiring classification only (no order lookup)
    # -----------------------------------------------------------------------
    def test_j_classification_only_ticket(self):
        mock_client = MagicMock()

        round1 = _make_response(
            function_calls=[
                _make_call(
                    "classify_ticket_category",
                    {"text": "Mobile app crashes at checkout with fatal error"},
                    "c1",
                )
            ]
        )
        final_json = json.dumps(
            {
                "ticket_id": "T007",
                "category": "bug",
                "tools_called": ["classify_ticket_category"],
                "draft_reply": (
                    "Thank you for reporting this mobile checkout crash. We have logged "
                    "the issue for our engineering team to investigate."
                ),
            }
        )
        round2 = _make_response(function_calls=None, text=final_json)
        mock_client.models.generate_content.side_effect = [round1, round2]

        ticket = {
            "ticket_id": "T007",
            "subject": "Mobile app crashes at checkout",
            "body": "The mobile app freezes and crashes with a fatal error.",
            "customer_email": "tech.user@example.com",
            "order_id": None,
        }

        result = triage_ticket(ticket, mock_client, verbose=False)

        self.assertIsNone(result["error"])
        self.assertEqual(result["category"], "bug")
        self.assertEqual(result["tools_called"], ["classify_ticket_category"])
        self.assertNotIn("lookup_order_status", result["tools_called"])
        self.assertEqual(result["tool_rounds"], 1)

    # -----------------------------------------------------------------------
    # Test L: Malformed tool arguments, unknown tools, and invalid responses
    # -----------------------------------------------------------------------
    def test_l_malformed_arguments_unknown_tools_and_invalid_responses(self):
        cache = {}

        # 1. Unknown tool name
        res_unknown, executed = validate_and_execute_tool(
            _make_call("drop_database_table", {"table": "orders"}), cache
        )
        self.assertFalse(executed)
        self.assertFalse(res_unknown["success"])
        self.assertIn("Unknown or unregistered tool", res_unknown["error"])

        # 2. Malformed non-dict args
        res_bad_type, executed = validate_and_execute_tool(
            _make_call("lookup_order_status", "not_a_dict"), cache
        )
        self.assertFalse(executed)
        self.assertIn("Malformed arguments", res_bad_type["error"])

        # 3. Unexpected keyword arguments
        res_unexpected, executed = validate_and_execute_tool(
            _make_call("lookup_order_status", {"order_id": "A123", "hack": True}), cache
        )
        self.assertFalse(executed)
        self.assertIn("Invalid argument(s)", res_unexpected["error"])

        # 4. Missing required argument
        res_missing, executed = validate_and_execute_tool(
            _make_call("lookup_order_status", {}), cache
        )
        self.assertFalse(executed)
        self.assertIn("Missing required argument(s)", res_missing["error"])

        # 5. Duplicate identical call prevention
        first_res, first_exec = validate_and_execute_tool(
            _make_call("lookup_order_status", {"order_id": "A123"}), cache
        )
        second_res, second_exec = validate_and_execute_tool(
            _make_call("lookup_order_status", {"order_id": "A123"}), cache
        )
        self.assertTrue(first_exec)
        self.assertFalse(second_exec)
        self.assertTrue(second_res.get("duplicate_call_prevented"))

        # 6. Malformed non-JSON final model response -> graceful fallback + review flag
        validated = validate_final_output(
            raw_text="Not valid JSON at all, plain apology text.",
            ticket_id="T999",
            executed_tools=["classify_ticket_category"],
            tool_results=[{"tool": "classify_ticket_category", "result": {"category": "shipping"}}],
        )
        self.assertTrue(validated["malformed_output"])
        self.assertTrue(validated["requires_human_review"])
        self.assertEqual(validated["category"], "shipping")
        self.assertIn("plain apology text", validated["draft_reply"])

        # 7. Empty final text response
        validated_empty = validate_final_output(
            raw_text="",
            ticket_id="T999",
            executed_tools=[],
            tool_results=[],
        )
        self.assertTrue(validated_empty["malformed_output"])
        self.assertTrue(validated_empty["requires_human_review"])
        self.assertEqual(validated_empty["category"], "other")
        self.assertTrue(len(validated_empty["draft_reply"]) > 10)

    def test_week1_process_user_query_compatibility(self):
        """Confirm Week 1 OrderPilot function-calling loop still works intact."""
        mock_client = MagicMock()
        r1 = _make_response(function_calls=[_make_call("lookup_order_status", {"order_id": "A123"})])
        r2 = _make_response(function_calls=None, text="Order A123 is Shipped and arrives Tomorrow.")
        mock_client.models.generate_content.side_effect = [r1, r2]

        ans = process_user_query("Where is A123?", mock_client, "gemini-3.5-flash", verbose=False)
        self.assertIn("Shipped", ans)
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    def test_regression_t004_invoice_discrepancy_excludes_refund_check(self):
        """Regression test: T004 invoice error must classify, lookup, escalate without refund check."""
        # Verify updated prompt & tool schema contain explicit boundary rules
        self.assertIn("ONLY when the customer explicitly asks for a refund", check_refund_eligibility_declaration.description)
        self.assertIn("Do NOT call `check_refund_eligibility` for general", TICKET_SYSTEM_INSTRUCTION)

        mock_client = MagicMock()
        r1 = _make_response(
            function_calls=[
                _make_call(
                    "classify_ticket_category",
                    {"text": "Incorrect invoice amount billed shows $20 more than checkout total."},
                    "c1",
                ),
                _make_call("lookup_order_status", {"order_id": "ORD-102"}, "c2"),
            ]
        )
        r2 = _make_response(
            function_calls=[
                _make_call(
                    "escalate_to_human",
                    {"reason": "$20 invoice overcharge on order ORD-102 requires billing review.", "ticket_id": "T004"},
                    "c3",
                )
            ]
        )
        r3 = _make_response(
            function_calls=None,
            text=json.dumps(
                {
                    "ticket_id": "T004",
                    "category": "billing",
                    "tools_called": ["classify_ticket_category", "lookup_order_status", "escalate_to_human"],
                    "draft_reply": "Hello Bob, we have queued your $20 invoice discrepancy on ORD-102 for billing review.",
                }
            ),
        )
        mock_client.models.generate_content.side_effect = [r1, r2, r3]

        ticket = {
            "ticket_id": "T004",
            "subject": "Incorrect invoice amount billed",
            "body": "I just checked my invoice for order ORD-102 and the transaction shows $20 more than the checkout total.",
            "customer_email": "bob.smith@example.com",
            "order_id": "ORD-102",
        }

        result = triage_ticket(ticket, mock_client, verbose=False)
        self.assertEqual(result["category"], "billing")
        self.assertTrue(result["requires_human_review"])
        self.assertNotIn("check_refund_eligibility", result["tools_called"])
        self.assertEqual(
            result["tools_called"],
            ["classify_ticket_category", "lookup_order_status", "escalate_to_human"],
        )


if __name__ == "__main__":
    unittest.main()
