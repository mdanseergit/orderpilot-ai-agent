"""
main.py - TicketPilot & OrderPilot CLI Entry Point

Provides:
1. Week 2 TicketPilot Batch Triage (`python main.py`):
   Reads all support tickets from `tickets.json`, runs the multi-round Gemini
   tool-calling agent, prints per-ticket execution details, and outputs a summary.
2. Preserved Week 1 OrderPilot Modes:
   - Interactive order-support chat: `python main.py --interactive`
   - Single order query: `python main.py --query "What's the status of order A123?"`
"""

import argparse
from collections import Counter
import json
import os
import sys
import time
from typing import Any, Dict, List
from google import genai
from google.genai import errors, types

from agent import MAX_TOOL_ROUNDS, _call_gemini_with_retry, load_gemini_client, triage_ticket
from tools import ORDER_TOOL, TOOL_REGISTRY, get_escalation_count, reset_escalation_log


# ---------------------------------------------------------------------------
# Preserved Week 1 OrderPilot Functionality
# ---------------------------------------------------------------------------
WEEK1_SYSTEM_INSTRUCTION = (
    "You are OrderPilot, an AI order-support assistant for an e-commerce platform.\n"
    "Your responsibilities:\n"
    "1. When a user inquires about an order's status, delivery date, or shipping, "
    "extract the order ID and invoke the 'lookup_order_status' tool.\n"
    "2. If the user asks about an order but does not mention an order ID, politely ask them "
    "to provide their order ID. Never invent, guess, or assume an order ID.\n"
    "3. Base your final response strictly and accurately on the structured result returned by the tool.\n"
    "4. If an order is not found, clearly and politely inform the customer that the order was not found.\n"
    "5. For non-order inquiries (greetings, questions about your capabilities, etc.), "
    "respond directly and helpfully without calling any tools."
)


def execute_tool_call(call: types.FunctionCall) -> Dict[str, Any]:
    """Week 1 helper: Match and execute a model-requested function against TOOL_REGISTRY."""
    func_name = call.name
    raw_args = call.args or {}

    if func_name not in TOOL_REGISTRY:
        return {
            "found": False,
            "error": f"Unknown tool '{func_name}'. Not in registered tools.",
        }

    target_func = TOOL_REGISTRY[func_name]

    try:
        clean_args = {k: v for k, v in raw_args.items()}
        return target_func(**clean_args)
    except TypeError as te:
        return {
            "found": False,
            "error": f"Invalid arguments supplied for '{func_name}': {str(te)}",
        }
    except Exception as exc:
        return {
            "found": False,
            "error": f"Unexpected error during execution of '{func_name}': {str(exc)}",
        }


def process_user_query(
    user_query: str,
    client: genai.Client,
    model_name: str,
    max_iterations: int = 5,
    verbose: bool = True,
) -> str:
    """Week 1 helper: Run single-tool Gemini function-calling loop for a customer query."""
    if verbose:
        print("\n[AI] Analyzing your request...")

    config = types.GenerateContentConfig(
        tools=[ORDER_TOOL],
        temperature=0.0,
        system_instruction=WEEK1_SYSTEM_INSTRUCTION,
    )

    contents: List[Any] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )
    ]

    for _ in range(1, max_iterations + 1):
        try:
            response = _call_gemini_with_retry(
                client=client,
                model_name=model_name,
                contents=contents,
                config=config,
                verbose=verbose,
            )
        except errors.APIError as api_err:
            error_msg = f"Gemini API Error: {api_err.message} (code: {api_err.code})"
            if verbose:
                print(f"\n[ERROR] {error_msg}")
            return error_msg
        except Exception as exc:
            error_msg = f"Unexpected communication error: {str(exc)}"
            if verbose:
                print(f"\n[ERROR] {error_msg}")
            return error_msg

        if not response.function_calls:
            final_text = response.text or "I was unable to generate a response. Please try again."
            return final_text.strip()

        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        tool_response_parts: List[types.Part] = []

        for call in response.function_calls:
            call_args = dict(call.args) if call.args else {}

            if verbose:
                print("\n[TOOL CALL]")
                print(f"Function:  {call.name}")
                print(f"Arguments: {json.dumps(call_args, indent=2)}")

            tool_result = execute_tool_call(call)

            if verbose:
                print("\n[TOOL EXECUTION]")
                print(f"Result:    {json.dumps(tool_result, indent=2)}")

            tool_part = types.Part.from_function_response(
                name=call.name,
                response=tool_result,
            )
            tool_response_parts.append(tool_part)

        contents.append(types.Content(role="user", parts=tool_response_parts))

    return "Reached maximum tool-call iterations without a final response."


def run_interactive_loop(client: genai.Client, model_name: str) -> None:
    """Week 1 interactive terminal CLI session."""
    print("=" * 65)
    print("        ORDERPILOT — AI ORDER SUPPORT AGENT (WEEK 1 MODE)")
    print(f"        Active Model: {model_name}")
    print("=" * 65)
    print("Ask about any order status (e.g., 'What's the status of order A123?')")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("Enter your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! Thank you for using OrderPilot.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("\nGoodbye! Thank you for using OrderPilot.")
            break

        final_answer = process_user_query(
            user_query=user_input,
            client=client,
            model_name=model_name,
            verbose=True,
        )

        print("\n[FINAL ANSWER]")
        print(final_answer)
        print("-" * 65)


# ---------------------------------------------------------------------------
# Week 2 TicketPilot Batch Triage Runner
# ---------------------------------------------------------------------------
def load_tickets(tickets_path: str) -> List[Dict[str, Any]]:
    """Load support tickets from JSON file."""
    if not os.path.exists(tickets_path):
        raise FileNotFoundError(f"Tickets file not found: {tickets_path}")

    with open(tickets_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("tickets.json must contain a JSON array of ticket objects.")
    return data


def run_ticket_triage(
    tickets_path: str = "tickets.json",
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    pace_seconds: float = 0.0,
) -> int:
    """Process all tickets in tickets.json and print per-ticket and summary reports."""
    try:
        client, model_name = load_gemini_client()
    except ValueError as err:
        print(f"\n[CONFIGURATION ERROR]\n{err}\n")
        return 1

    try:
        tickets = load_tickets(tickets_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as err:
        print(f"\n[DATA ERROR]\n{err}\n")
        return 1

    reset_escalation_log()

    print("=" * 75)
    print("           TICKETPILOT — MULTI-TOOL SUPPORT TRIAGE AGENT")
    print(f"           Model: {model_name} | Max Tool Rounds: {max_tool_rounds}")
    print(f"           Tickets Loaded: {len(tickets)}")
    print("=" * 75)

    success_count = 0
    failed_count = 0
    human_review_count = 0
    tool_invocation_counts: Counter = Counter()

    for idx, ticket in enumerate(tickets, start=1):
        ticket_id = ticket.get("ticket_id", f"UNKNOWN-{idx}")
        subject = ticket.get("subject", "(No Subject)")

        print(f"\n{'-' * 75}")
        print(f"[{idx}/{len(tickets)}] Ticket ID: {ticket_id} | Subject: {subject}")
        print(f"{'-' * 75}")

        result = triage_ticket(
            ticket=ticket,
            client=client,
            model_name=model_name,
            max_tool_rounds=max_tool_rounds,
            verbose=True,
        )

        # Count actual executed tool invocations
        for entry in result.get("tool_results", []):
            if entry.get("freshly_executed", True):
                tool_invocation_counts[entry.get("tool", "unknown")] += 1

        if result.get("requires_human_review"):
            human_review_count += 1

        if result.get("error"):
            failed_count += 1
        else:
            success_count += 1

        print("\n  [TRIAGE RESULT]")
        print(f"  Ticket ID:             {result['ticket_id']}")
        print(f"  Subject:               {result['subject']}")
        print(f"  Selected Category:     {result['category']}")
        print(f"  Tools Called:          {result['tools_called']}")
        print(f"  Tool Rounds Executed:  {result['tool_rounds']}")
        print(f"  Requires Human Review: {result['requires_human_review']}")
        if result.get("error"):
            print(f"  Error / Warning:       {result['error']}")
        print(f"  Final Draft Reply:\n    {result['draft_reply']}")

        if pace_seconds > 0 and idx < len(tickets):
            time.sleep(pace_seconds)

    escalation_count = get_escalation_count()

    print("\n" + "=" * 75)
    print("                     TICKETPILOT TRIAGE SUMMARY")
    print("=" * 75)
    print(f"Total Tickets Processed:   {len(tickets)}")
    print(f"Successful Triage Count:   {success_count}")
    print(f"Failed / Warned Count:     {failed_count}")
    print(f"Human Review Required:     {human_review_count}")
    print(f"Simulated Escalations:     {escalation_count}")
    print("Tool Invocation Counts:")
    if tool_invocation_counts:
        for tool_name, count in sorted(tool_invocation_counts.items()):
            print(f"  - {tool_name}: {count}")
    else:
        print("  - (none)")
    print("=" * 75)

    return 0 if failed_count == 0 else 1


def main() -> None:
    """CLI entry point supporting Week 2 TicketPilot batch mode and Week 1 OrderPilot modes."""
    parser = argparse.ArgumentParser(
        description="TicketPilot — Multi-Tool AI Support Ticket Triage Agent (with Week 1 OrderPilot compatibility)"
    )
    parser.add_argument(
        "--tickets",
        default="tickets.json",
        help="Path to JSON file of support tickets (default: tickets.json)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_TOOL_ROUNDS,
        help=f"Maximum tool-calling rounds per ticket (default: {MAX_TOOL_ROUNDS})",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=0.0,
        help="Optional pause in seconds between tickets to respect API rate limits.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch Week 1 OrderPilot interactive chat session.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single Week 1 OrderPilot question from the command line.",
    )

    args = parser.parse_args()

    if args.interactive or args.query:
        try:
            client, model_name = load_gemini_client()
        except ValueError as err:
            print(f"\n[CONFIGURATION ERROR]\n{err}\n")
            sys.exit(1)

        if args.query:
            print(f"Question: {args.query}")
            ans = process_user_query(args.query, client, model_name, verbose=True)
            print("\n[FINAL ANSWER]")
            print(ans)
        else:
            run_interactive_loop(client, model_name)
        return

    exit_code = run_ticket_triage(
        tickets_path=args.tickets,
        max_tool_rounds=args.max_rounds,
        pace_seconds=args.pace,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

