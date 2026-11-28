"""
test_tools.py - Offline Unit Tests for TicketPilot & Week 1 OrderPilot Tools

Covers:
- Test A: Valid order lookup (including Week 1 compatibility).
- Test B: Unknown order lookup.
- Test C: Billing category classification.
- Test D: Shipping category classification.
- Test E: Bug category classification.
- Test F: Refund eligibility for a delivered order within 30 days.
- Test G: Refund eligibility for a shipped order.
- Test H: Unknown order refund check.
- Additional: Other/empty classification, overlapping keyword precedence,
  expired (>30 days) delivery, cancelled order, and simulated escalation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import (
    TOOL_REGISTRY,
    check_refund_eligibility,
    classify_ticket_category,
    escalate_to_human,
    get_escalation_count,
    lookup_order_status,
    reset_escalation_log,
)


class TestTicketPilotTools(unittest.TestCase):
    """Deterministic, 100% offline unit tests for all TicketPilot Python tools."""

    def setUp(self):
        reset_escalation_log()

    # -----------------------------------------------------------------------
    # Test A: Valid Order Lookup
    # -----------------------------------------------------------------------
    def test_a_valid_order_lookup(self):
        res = lookup_order_status("ORD-101")
        self.assertTrue(res["found"])
        self.assertEqual(res["order_id"], "ORD-101")
        self.assertEqual(res["customer_name"], "Alice Johnson")
        self.assertEqual(res["product"], "Wireless Noise-Canceling Headphones")
        self.assertEqual(res["status"], "Shipped")
        self.assertEqual(res["delivery_date"], "Tomorrow by 2:00 PM")

        # Case-insensitive + Week 1 backward compatibility
        res_w1 = lookup_order_status("  a123  ")
        self.assertTrue(res_w1["found"])
        self.assertEqual(res_w1["order_id"], "A123")
        self.assertEqual(res_w1["status"], "Shipped")

    # -----------------------------------------------------------------------
    # Test B: Unknown Order Lookup
    # -----------------------------------------------------------------------
    def test_b_unknown_order_lookup(self):
        res = lookup_order_status("XYZ999")
        self.assertFalse(res["found"])
        self.assertEqual(res["order_id"], "XYZ999")
        self.assertIn("not found", res["message"].lower())

        # Invalid/empty order ID validation
        self.assertFalse(lookup_order_status("")["found"])
        self.assertFalse(lookup_order_status("   ")["found"])
        self.assertFalse(lookup_order_status(None)["found"])  # type: ignore[arg-type]

    # -----------------------------------------------------------------------
    # Test C: Billing Category Classification
    # -----------------------------------------------------------------------
    def test_c_billing_category_classification(self):
        res = classify_ticket_category(
            "Duplicate charge on my card. I was charged twice for the transaction on my invoice."
        )
        self.assertEqual(res["category"], "billing")
        self.assertIn("billing", res["matched_keywords"])

    # -----------------------------------------------------------------------
    # Test D: Shipping Category Classification
    # -----------------------------------------------------------------------
    def test_d_shipping_category_classification(self):
        res = classify_ticket_category(
            "Order delayed in transit. The courier tracking shows a missing package."
        )
        self.assertEqual(res["category"], "shipping")
        self.assertIn("shipping", res["matched_keywords"])

    # -----------------------------------------------------------------------
    # Test E: Bug Category Classification
    # -----------------------------------------------------------------------
    def test_e_bug_category_classification(self):
        res = classify_ticket_category(
            "Mobile app crashes with a fatal error and login failure when opening checkout."
        )
        self.assertEqual(res["category"], "bug")
        self.assertIn("bug", res["matched_keywords"])

    def test_other_and_overlapping_keyword_precedence(self):
        # No matching keywords -> 'other'
        res_other = classify_ticket_category("Do you offer gift wrapping next month?")
        self.assertEqual(res_other["category"], "other")

        # Empty / invalid text -> 'other'
        self.assertEqual(classify_ticket_category("")["category"], "other")
        self.assertEqual(classify_ticket_category(None)["category"], "other")  # type: ignore[arg-type]

        # Overlapping tie (1 billing term 'charged' vs 1 shipping term 'delivery')
        # Precedence: billing > shipping > bug
        res_tie = classify_ticket_category("I was charged extra for delivery.")
        self.assertEqual(res_tie["category"], "billing")
        self.assertIn("precedence", res_tie["reason"].lower())

    # -----------------------------------------------------------------------
    # Test F: Refund Eligibility for Delivered Order Within 30 Days
    # -----------------------------------------------------------------------
    def test_f_refund_eligibility_delivered_within_30_days(self):
        # ORD-103 was delivered 2026-11-25; simulated today is 2026-11-28 (3 days ago)
        res = check_refund_eligibility("ORD-103", current_date="2026-11-28")
        self.assertTrue(res["found"])
        self.assertTrue(res["eligible"])
        self.assertFalse(res["requires_human_review"])
        self.assertFalse(res["refund_issued"])
        self.assertEqual(res["policy_rule"], "delivered_within_30_days")

    def test_refund_eligibility_delivered_outside_30_days_and_cancelled(self):
        # ORD-107 delivered 2026-10-10 (49 days before 2026-11-28) -> ineligible
        res_old = check_refund_eligibility("ORD-107", current_date="2026-11-28")
        self.assertTrue(res_old["found"])
        self.assertFalse(res_old["eligible"])
        self.assertEqual(res_old["policy_rule"], "delivered_outside_30_days")

        # ORD-105 Cancelled -> eligible
        res_cancelled = check_refund_eligibility("ORD-105")
        self.assertTrue(res_cancelled["found"])
        self.assertTrue(res_cancelled["eligible"])
        self.assertFalse(res_cancelled["refund_issued"])
        self.assertEqual(res_cancelled["policy_rule"], "cancelled_order")

    # -----------------------------------------------------------------------
    # Test G: Refund Eligibility for Shipped (and Processing) Order
    # -----------------------------------------------------------------------
    def test_g_refund_eligibility_shipped_order(self):
        res = check_refund_eligibility("ORD-101")
        self.assertTrue(res["found"])
        self.assertFalse(res["eligible"])
        self.assertTrue(res["requires_human_review"])
        self.assertFalse(res["refund_issued"])
        self.assertEqual(res["policy_rule"], "in_progress_requires_review")

        res_proc = check_refund_eligibility("ORD-102")
        self.assertFalse(res_proc["eligible"])
        self.assertTrue(res_proc["requires_human_review"])

    # -----------------------------------------------------------------------
    # Test H: Unknown Order Refund Check
    # -----------------------------------------------------------------------
    def test_h_unknown_order_refund_check(self):
        res = check_refund_eligibility("ORD-999")
        self.assertFalse(res["found"])
        self.assertFalse(res["eligible"])
        self.assertFalse(res["refund_issued"])
        self.assertEqual(res["policy_rule"], "unknown_order")

        res_empty = check_refund_eligibility("")
        self.assertFalse(res_empty["eligible"])
        self.assertEqual(res_empty["policy_rule"], "invalid_order_id")

    # -----------------------------------------------------------------------
    # Optional Stretch Goal: Simulated Escalation Tool
    # -----------------------------------------------------------------------
    def test_escalate_to_human_tool(self):
        self.assertEqual(get_escalation_count(), 0)
        res = escalate_to_human("Suspected unauthorized transaction on ORD-999", ticket_id="T012")
        self.assertTrue(res["escalated"])
        self.assertTrue(res["simulated"])
        self.assertEqual(res["ticket_id"], "T012")
        self.assertEqual(get_escalation_count(), 1)
        self.assertIn("no live human notification", res["note"].lower())

        # Empty reason rejected
        bad = escalate_to_human("   ")
        self.assertFalse(bad["escalated"])
        self.assertEqual(get_escalation_count(), 1)

    def test_tool_registry_completeness(self):
        for name in (
            "lookup_order_status",
            "classify_ticket_category",
            "check_refund_eligibility",
            "escalate_to_human",
        ):
            self.assertIn(name, TOOL_REGISTRY)


if __name__ == "__main__":
    unittest.main()
