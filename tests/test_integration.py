"""
test_integration.py - Live Gemini Integration Tests (Week 1 + Week 2)

Enabled only when GEMINI_API_KEY is set in the environment or .env file.
Runs real calls against Google's Gemini API to verify:
1. Week 2 multi-tool ticket triage (refund ticket calling classify + lookup + refund check).
2. Week 2 classification-only ticket (bug report without order_id).
3. Week 1 OrderPilot single-tool query compatibility.
"""

import os
import sys
import time
import unittest
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent import load_gemini_client, triage_ticket
from main import process_user_query


load_dotenv()
HAS_API_KEY = bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY").strip())


@unittest.skipUnless(HAS_API_KEY, "GEMINI_API_KEY not configured; skipping live integration tests.")
class TestLiveGeminiIntegration(unittest.TestCase):
    """Live API integration tests verifying real Gemini tool calling."""

    @classmethod
    def setUpClass(cls):
        cls.client, cls.model_name = load_gemini_client()

    def test_live_ticketpilot_multi_tool_refund(self):
        ticket = {
            "ticket_id": "LIVE-T003",
            "subject": "Refund for cancelled hub order",
            "body": "My order ORD-105 was cancelled earlier this week. Am I eligible for a refund?",
            "customer_email": "emma.watson@example.com",
            "order_id": "ORD-105",
        }
        result = triage_ticket(ticket, self.client, self.model_name, verbose=False)
        self.assertIsNone(result["error"])
        self.assertEqual(result["category"], "billing")
        self.assertIn("classify_ticket_category", result["tools_called"])
        self.assertIn("lookup_order_status", result["tools_called"])
        self.assertIn("check_refund_eligibility", result["tools_called"])
        self.assertIn("refund", result["draft_reply"].lower())

    def test_live_ticketpilot_classification_only(self):
        time.sleep(2)
        ticket = {
            "ticket_id": "LIVE-T007",
            "subject": "Mobile app crashes at checkout",
            "body": "The mobile app freezes and crashes with a fatal error during checkout.",
            "customer_email": "tech.user@example.com",
            "order_id": None,
        }
        result = triage_ticket(ticket, self.client, self.model_name, verbose=False)
        self.assertIsNone(result["error"])
        self.assertEqual(result["category"], "bug")
        self.assertIn("classify_ticket_category", result["tools_called"])
        self.assertNotIn("lookup_order_status", result["tools_called"])

    def test_live_week1_orderpilot_lookup(self):
        time.sleep(2)
        answer = process_user_query(
            "What's the status of order ORD-101?",
            self.client,
            self.model_name,
            verbose=False,
        )
        self.assertTrue("shipped" in answer.lower() or "tomorrow" in answer.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
