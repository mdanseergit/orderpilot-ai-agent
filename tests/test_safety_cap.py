"""
test_safety_cap.py - Safety-Cap Verification Tests

Covers:
- Test K: Deliberately controlled mock model that keeps requesting tool calls
  indefinitely. Proves execution stops at exactly MAX_TOOL_ROUNDS (and any
  custom configured limit), logs a clear warning, marks requires_human_review=True,
  and never fabricates a successful result.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent import MAX_TOOL_ROUNDS, triage_ticket


class TestSafetyCap(unittest.TestCase):
    """Tests proving the tool-calling loop halts cleanly at MAX_TOOL_ROUNDS."""

    @staticmethod
    def _infinite_tool_responses():
        """Generator simulating a rogue model that endlessly requests new tool calls."""
        counter = 0
        while True:
            counter += 1
            call = MagicMock()
            call.name = "lookup_order_status"
            # Vary order_id so each round executes freshly without cache short-circuiting
            call.args = {"order_id": f"UNKNOWN-{counter}"}
            call.id = f"call_{counter}"

            resp = MagicMock()
            resp.function_calls = [call]
            resp.text = None
            candidate = MagicMock()
            candidate.content = MagicMock()
            resp.candidates = [candidate]
            yield resp

    # -----------------------------------------------------------------------
    # Test K: Safety Cap at Default MAX_TOOL_ROUNDS (5)
    # -----------------------------------------------------------------------
    def test_k_safety_cap_stops_at_default_max_rounds(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = self._infinite_tool_responses()

        ticket = {
            "ticket_id": "T-CAP-5",
            "subject": "Runaway tool loop test",
            "body": "Test infinite tool call protection.",
            "customer_email": "test@example.com",
            "order_id": "ORD-101",
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = triage_ticket(
                ticket=ticket,
                client=mock_client,
                model_name="gemini-3.5-flash",
                max_tool_rounds=MAX_TOOL_ROUNDS,
                verbose=True,
            )

        output_logs = buf.getvalue()

        # Exactly 5 tool rounds executed; 6th model call requested tools and was halted
        self.assertEqual(MAX_TOOL_ROUNDS, 5)
        self.assertEqual(result["tool_rounds"], 5)
        self.assertEqual(len(result["tool_results"]), 5)
        self.assertEqual(mock_client.models.generate_content.call_count, 6)

        # Must flag safety cap, require human review, log warning, and not fabricate success
        self.assertTrue(result["safety_cap_reached"])
        self.assertTrue(result["requires_human_review"])
        self.assertIsNotNone(result["error"])
        self.assertIn("SAFETY CAP REACHED", result["error"])
        self.assertIn("[WARNING] SAFETY CAP REACHED", output_logs)
        self.assertIn("human review", result["draft_reply"].lower())

    def test_safety_cap_custom_configurable_limit(self):
        """Verify custom max_tool_rounds values (e.g., 2) are strictly enforced."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = self._infinite_tool_responses()

        ticket = {
            "ticket_id": "T-CAP-2",
            "subject": "Custom cap test",
            "body": "Test cap of 2 rounds.",
            "customer_email": "test@example.com",
            "order_id": None,
        }

        result = triage_ticket(
            ticket=ticket,
            client=mock_client,
            max_tool_rounds=2,
            verbose=False,
        )

        self.assertEqual(result["tool_rounds"], 2)
        self.assertEqual(len(result["tool_results"]), 2)
        self.assertTrue(result["safety_cap_reached"])
        self.assertTrue(result["requires_human_review"])
        self.assertEqual(mock_client.models.generate_content.call_count, 3)


if __name__ == "__main__":
    unittest.main()
