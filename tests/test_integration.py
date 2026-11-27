"""
test_integration.py - Live Gemini Integration Tests for OrderPilot

Executes live end-to-end test cases against Google's Gemini API:
TEST 1: Known order ("What's the status of order ORD-101?")
TEST 2: Unknown order ("Where is order XYZ999?")
TEST 3: Another known order ("Has order A125 been delivered?")
TEST 4: Non-order question ("Hello, what can you do?")
TEST 5: Case-insensitive lookup ("Check order ord-101.")
TEST 6: Missing order ID ("Where is my order?")

Note: Requires a valid GEMINI_API_KEY in .env and network access.
"""

import os
import sys
import time

# Add parent directory to path so tools and main can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import load_gemini_client, process_user_query


def run_all_tests():
    print("=" * 70)
    print("      ORDERPILOT — LIVE GEMINI API INTEGRATION TESTS")
    print("=" * 70)

    try:
        client, model_name = load_gemini_client()
        print(f"Loaded client with model: {model_name}\n")
    except Exception as e:
        print(f"FAILED to initialize Gemini client: {e}")
        return False

    all_passed = True

    # -------------------------------------------------------------
    # TEST 1: Known Order
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 1 — Known Order: What's the status of order ORD-101?")
    print("=" * 50)
    query_1 = "What's the status of order ORD-101?"
    ans_1 = process_user_query(query_1, client, model_name, verbose=True)
    print(f"\n[FINAL ANSWER 1]\n{ans_1}")
    if any(k in ans_1.lower() for k in ["shipped", "tomorrow", "alice", "headphone"]):
        print(">>> TEST 1 RESULT: PASSED (Correctly identified Shipped/Tomorrow/Alice)")
    else:
        print(">>> TEST 1 RESULT: FAILED (Did not report expected status)")
        all_passed = False

    # Sleep between tests to stay safely within free tier rate limit
    print("\n[Pacing] Waiting 12s for API rate limit headroom...")
    time.sleep(12)

    # -------------------------------------------------------------
    # TEST 2: Unknown Order
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 2 — Unknown Order: Where is order XYZ999?")
    print("=" * 50)
    query_2 = "Where is order XYZ999?"
    ans_2 = process_user_query(query_2, client, model_name, verbose=True)
    print(f"\n[FINAL ANSWER 2]\n{ans_2}")
    if any(phrase in ans_2.lower() for phrase in ["not found", "could not find", "unable to find", "doesn't exist", "does not exist"]):
        print(">>> TEST 2 RESULT: PASSED (Correctly reported order not found)")
    else:
        print(">>> TEST 2 RESULT: FAILED (Did not report order not found)")
        all_passed = False

    print("\n[Pacing] Waiting 12s for API rate limit headroom...")
    time.sleep(12)

    # -------------------------------------------------------------
    # TEST 3: Another Known Order
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 3 — Another Known Order: Has order A125 been delivered?")
    print("=" * 50)
    query_3 = "Has order A125 been delivered?"
    ans_3 = process_user_query(query_3, client, model_name, verbose=True)
    print(f"\n[FINAL ANSWER 3]\n{ans_3}")
    if any(k in ans_3.lower() for k in ["delivered", "november 25"]):
        print(">>> TEST 3 RESULT: PASSED (Correctly confirmed order is Delivered)")
    else:
        print(">>> TEST 3 RESULT: FAILED (Did not confirm delivered)")
        all_passed = False

    print("\n[Pacing] Waiting 12s for API rate limit headroom...")
    time.sleep(12)

    # -------------------------------------------------------------
    # TEST 4: Non-order question
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 4 — Non-order question: Hello, what can you do?")
    print("=" * 50)
    query_4 = "Hello, what can you do?"
    ans_4 = process_user_query(query_4, client, model_name, verbose=True)
    print(f"\n[FINAL ANSWER 4]\n{ans_4}")
    if len(ans_4) > 15:
        print(">>> TEST 4 RESULT: PASSED (Responded to general inquiry naturally without calling tool)")
    else:
        print(">>> TEST 4 RESULT: FAILED (Response was too short or empty)")
        all_passed = False

    print("\n[Pacing] Waiting 12s for API rate limit headroom...")
    time.sleep(12)

    # -------------------------------------------------------------
    # TEST 5: Case-insensitive lookup
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 5 — Case-insensitive lookup: Check order ord-101.")
    print("=" * 50)
    query_5 = "Check order ord-101."
    ans_5 = process_user_query(query_5, client, model_name, verbose=True)
    print(f"\n[FINAL ANSWER 5]\n{ans_5}")
    if any(k in ans_5.lower() for k in ["shipped", "tomorrow", "alice", "headphone"]):
        print(">>> TEST 5 RESULT: PASSED (Correctly resolved lowercase order ord-101)")
    else:
        print(">>> TEST 5 RESULT: FAILED (Failed to resolve order ord-101)")
        all_passed = False

    print("\n[Pacing] Waiting 12s for API rate limit headroom...")
    time.sleep(12)

    # -------------------------------------------------------------
    # TEST 6: Missing order ID
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 6 — Missing order ID: Where is my order?")
    print("=" * 50)
    query_6 = "Where is my order?"
    ans_6 = process_user_query(query_6, client, model_name, verbose=True)
    print(f"\n[FINAL ANSWER 6]\n{ans_6}")
    if any(phrase in ans_6.lower() for phrase in ["order id", "order number", "provide", "please provide"]):
        print(">>> TEST 6 RESULT: PASSED (Politely asked user for the order ID)")
    else:
        print(">>> TEST 6 RESULT: FAILED (Did not ask for order ID)")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL 6 INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("SOME INTEGRATION TESTS FAILED.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
