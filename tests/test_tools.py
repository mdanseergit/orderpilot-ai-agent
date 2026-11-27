"""
test_tools.py - Offline Unit Tests for OrderPilot

All tests in this file run 100% offline without requiring API keys or network access.
Gemini API calls are fully mocked to ensure fast, deterministic CI/CD execution.

Test Cases:
- Test 1: Valid order ID lookup (ORD-101, A123, case-insensitivity).
- Test 2: Unknown order ID handling (graceful error response).
- Test 3: Invalid or empty order ID validation (None, numbers, blanks).
- Test 4: Multiple independent order lookups.
- Test 5: Gemini tool-calling integration (mocked end-to-end loop).
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import ORDERS
from tools import TOOL_REGISTRY, lookup_order_status, lookup_order_status_declaration
from main import execute_tool_call, process_user_query


class TestOrderPilot(unittest.TestCase):
    """Offline unit test suite for OrderPilot."""

    # -----------------------------------------------------------------------
    # TEST 1: Valid Order ID
    # -----------------------------------------------------------------------
    def test_1_valid_order_id(self):
        """Verify valid order IDs return accurate, structured order data."""
        res_101 = lookup_order_status("ORD-101")
        self.assertTrue(res_101["found"])
        self.assertEqual(res_101["order_id"], "ORD-101")
        self.assertEqual(res_101["customer_name"], "Alice Johnson")
        self.assertEqual(res_101["product"], "Wireless Noise-Canceling Headphones")
        self.assertEqual(res_101["status"], "Shipped")
        self.assertEqual(res_101["delivery_date"], "Tomorrow by 2:00 PM")

        # Test case-insensitivity and whitespace handling
        res_lower = lookup_order_status("ord-101")
        self.assertTrue(res_lower["found"])
        self.assertEqual(res_lower["order_id"], "ORD-101")

        res_spaces = lookup_order_status("   A123   ")
        self.assertTrue(res_spaces["found"])
        self.assertEqual(res_spaces["status"], "Shipped")

    # -----------------------------------------------------------------------
    # TEST 2: Unknown Order ID
    # -----------------------------------------------------------------------
    def test_2_unknown_order_id(self):
        """Verify unknown order IDs return a structured not-found response without crashing."""
        result = lookup_order_status("XYZ999")
        self.assertFalse(result["found"])
        self.assertEqual(result["order_id"], "XYZ999")
        self.assertIn("message", result)
        self.assertIn("not found", result["message"].lower())

    # -----------------------------------------------------------------------
    # TEST 3: Invalid or Empty Order ID
    # -----------------------------------------------------------------------
    def test_3_invalid_or_empty_order_id(self):
        """Verify empty strings, blanks, and non-string inputs are properly validated."""
        res_empty = lookup_order_status("")
        self.assertFalse(res_empty["found"])
        self.assertIn("cannot be empty", res_empty["message"].lower())

        res_blank = lookup_order_status("     ")
        self.assertFalse(res_blank["found"])
        self.assertIn("cannot be empty", res_blank["message"].lower())

        res_none = lookup_order_status(None)  # type: ignore[arg-type]
        self.assertFalse(res_none["found"])
        self.assertIn("expected a string", res_none["message"].lower())

        res_int = lookup_order_status(98765)  # type: ignore[arg-type]
        self.assertFalse(res_int["found"])
        self.assertIn("expected a string", res_int["message"].lower())

    # -----------------------------------------------------------------------
    # TEST 4: Multiple Independent Order Lookups
    # -----------------------------------------------------------------------
    def test_4_multiple_order_lookups(self):
        """Verify multiple consecutive lookups return independent, correct records."""
        test_ids = ["ORD-101", "ORD-102", "ORD-103", "ORD-104", "ORD-105"]
        expected_statuses = {
            "ORD-101": "Shipped",
            "ORD-102": "Processing",
            "ORD-103": "Delivered",
            "ORD-104": "Out for delivery",
            "ORD-105": "Cancelled",
        }

        for order_id in test_ids:
            res = lookup_order_status(order_id)
            self.assertTrue(res["found"])
            self.assertEqual(res["status"], expected_statuses[order_id])
            self.assertEqual(res["order_id"], order_id)

    # -----------------------------------------------------------------------
    # TEST 5: Mocked Gemini Tool-Calling Integration
    # -----------------------------------------------------------------------
    def test_5_gemini_tool_calling_integration_mocked(self):
        """Verify end-to-end loop: model requests tool, Python executes, model responds."""
        mock_client = MagicMock()

        # Create a mock FunctionCall object
        mock_call = MagicMock()
        mock_call.name = "lookup_order_status"
        mock_call.args = {"order_id": "ORD-101"}

        # Candidate content for model's turn
        mock_candidate_content = MagicMock()

        # Turn 1 Response: Gemini decides to invoke lookup_order_status
        turn_1_resp = MagicMock()
        turn_1_resp.function_calls = [mock_call]
        turn_1_resp.candidates = [MagicMock(content=mock_candidate_content)]
        turn_1_resp.text = None

        # Turn 2 Response: Gemini produces the final natural-language response
        turn_2_resp = MagicMock()
        turn_2_resp.function_calls = None
        turn_2_resp.text = (
            "Your order ORD-101 for Alice Johnson (Wireless Noise-Canceling Headphones) "
            "is Shipped and will arrive Tomorrow by 2:00 PM."
        )

        # Configure client mock to return turn 1 then turn 2
        mock_client.models.generate_content.side_effect = [turn_1_resp, turn_2_resp]

        user_query = "What is the status of order ORD-101?"
        final_answer = process_user_query(
            user_query=user_query,
            client=mock_client,
            model_name="gemini-3.5-flash",
            verbose=False,
        )

        # Verify final answer matches turn 2 response
        self.assertEqual(final_answer, turn_2_resp.text)

        # Verify generate_content was called exactly twice (prompt + tool result)
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    def test_execute_tool_call_safety(self):
        """Verify execute_tool_call guards against unregistered tool execution."""
        mock_bad_call = MagicMock()
        mock_bad_call.name = "unregistered_dangerous_func"
        mock_bad_call.args = {}

        result = execute_tool_call(mock_bad_call)
        self.assertFalse(result["found"])
        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])

    def test_schema_and_registry(self):
        """Verify function declaration schema matches registration specifications."""
        self.assertEqual(lookup_order_status_declaration.name, "lookup_order_status")
        params = lookup_order_status_declaration.parameters_json_schema
        self.assertIsNotNone(params)
        self.assertIn("order_id", params["properties"])
        self.assertIn("order_id", params["required"])
        self.assertIn("lookup_order_status", TOOL_REGISTRY)


if __name__ == "__main__":
    unittest.main()
