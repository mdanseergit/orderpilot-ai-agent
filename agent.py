"""
agent.py - TicketPilot Multi-Round Tool-Calling Agent

Implements the core Gemini triage loop supporting:
- Multi-round function calling with configurable safety cap (MAX_TOOL_ROUNDS = 5).
- Multiple parallel tool calls within a single model response.
- Exact function-call ID/name matching for FunctionResponse turns.
- Strict argument validation, duplicate execution prevention, and API retry handling.
- Structured JSON output validation with safe fallbacks.
"""

import inspect
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from tools import TICKET_TOOLS, TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_TOOL_ROUNDS = 5
VALID_CATEGORIES = {"billing", "shipping", "bug", "other"}

TICKET_SYSTEM_INSTRUCTION = (
    "You are TicketPilot, an AI customer support triage specialist.\n"
    "Your job is to inspect each incoming support ticket, run the appropriate Python tools, "
    "determine the correct support category, and draft a concise, polite customer-facing reply.\n\n"
    "STRICT OPERATING RULES:\n"
    "1. ALWAYS call `classify_ticket_category` using the ticket's subject and body to determine the category.\n"
    "2. Whenever an `order_id` is provided in the ticket (and is not null/empty) AND order details are relevant "
    "(shipping, delivery, billing on an order, product defects, or refunds), call `lookup_order_status`.\n"
    "3. Call `check_refund_eligibility` ONLY when the customer explicitly asks for a refund, return, or refund eligibility "
    "on an order (and an `order_id` is present). If they do, you MUST call BOTH `lookup_order_status` AND "
    "`check_refund_eligibility` before drafting your response. Do NOT call `check_refund_eligibility` for general "
    "invoice discrepancies or duplicate-charge investigations where an order refund was not requested.\n"
    "4. For non-order technical/general tickets (e.g., mobile app crash, login failure, general store inquiry without an order_id), "
    "call `classify_ticket_category` only and do NOT call order or refund tools.\n"
    "5. If a case involves suspected fraud, an unauthorized charge, an invoice overcharge or duplicate payment, "
    "a missing delivered package, an unknown order dispute, or a refund request requiring human approval "
    "(such as Processing/Shipped/Out for delivery orders), call `escalate_to_human` with a clear reason.\n"
    "6. NEVER invent or fabricate order statuses, dates, or refund decisions.\n"
    "7. NEVER claim that a refund has already been issued or processed.\n"
    "8. NEVER claim that a real human has already been contacted or notified.\n"
    "9. NEVER expose internal tool names, JSON payloads, or tool execution details inside `draft_reply`.\n"
    "10. If information is insufficient (e.g., unknown order ID or missing details), state clearly in `draft_reply` what needs verification.\n\n"
    "FINAL OUTPUT FORMAT:\n"
    "Once you have completed all required tool calls, respond ONLY with a valid JSON object matching this exact schema:\n"
    "{\n"
    '  "ticket_id": "<ticket_id>",\n'
    '  "category": "<billing | shipping | bug | other>",\n'
    '  "tools_called": ["<tool_1>", "<tool_2>"],\n'
    '  "draft_reply": "<concise, professional customer-facing reply>"\n'
    "}"
)


# ---------------------------------------------------------------------------
# Gemini Client Initialization
# ---------------------------------------------------------------------------
def load_gemini_client() -> Tuple[genai.Client, str]:
    """Load environment configuration and initialize the Google Gen AI client.

    Returns:
        Tuple[genai.Client, str]: Configured client and selected Gemini model name.

    Raises:
        ValueError: If GEMINI_API_KEY is missing or blank.
    """
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "ERROR: GEMINI_API_KEY not found in environment or .env file.\n"
            "Please configure GEMINI_API_KEY in .env (see .env.example)."
        )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1beta"},
    )
    return client, model_name


def _call_gemini_with_retry(
    client: genai.Client,
    model_name: str,
    contents: List[Any],
    config: types.GenerateContentConfig,
    max_retries: int = 3,
    verbose: bool = True,
) -> Any:
    """Call Gemini generate_content with automatic exponential backoff on 429 rate limits."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except errors.APIError as api_err:
            if getattr(api_err, "code", None) == 429 and attempt < max_retries:
                wait_time = 20 * attempt
                if verbose:
                    print(
                        f"  [RATE LIMIT] Quota reached. Waiting {wait_time}s "
                        f"before retry {attempt}/{max_retries}..."
                    )
                time.sleep(wait_time)
                continue
            raise


# ---------------------------------------------------------------------------
# Tool Validation & Execution
# ---------------------------------------------------------------------------
def validate_and_execute_tool(
    call: Any,
    executed_cache: Dict[Tuple[str, str], Dict[str, Any]],
    ticket_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Safely validate tool name/arguments, prevent duplicates, and execute.

    Args:
        call: Gemini FunctionCall object with `name`, `args`, and optional `id`.
        executed_cache: Per-ticket cache mapping (name, canonical_args) -> result.
        ticket_id: Optional ticket ID for context injection on escalation.

    Returns:
        Tuple[Dict[str, Any], bool]: (structured tool result, whether freshly executed).
    """
    func_name = getattr(call, "name", None)
    if not isinstance(func_name, str) or func_name not in TOOL_REGISTRY:
        return {
            "error": f"Unknown or unregistered tool '{func_name}'. Execution blocked.",
            "success": False,
        }, False

    raw_args = getattr(call, "args", None)
    if raw_args is None:
        raw_args = {}
    elif not isinstance(raw_args, dict):
        try:
            raw_args = dict(raw_args)
        except (TypeError, ValueError):
            return {
                "error": f"Malformed arguments for '{func_name}': expected key-value mapping.",
                "success": False,
            }, False

    target_func = TOOL_REGISTRY[func_name]
    sig = inspect.signature(target_func)

    # Filter/validate arguments against function signature
    clean_args: Dict[str, Any] = {}
    unexpected_args: List[str] = []
    for k, v in raw_args.items():
        if k in sig.parameters:
            clean_args[k] = v
        else:
            unexpected_args.append(k)

    if unexpected_args:
        return {
            "error": (
                f"Invalid argument(s) {unexpected_args} for '{func_name}'. "
                f"Expected parameters: {list(sig.parameters.keys())}."
            ),
            "success": False,
        }, False

    # Inject ticket_id into escalate_to_human if omitted by model
    if func_name == "escalate_to_human" and "ticket_id" not in clean_args and ticket_id:
        clean_args["ticket_id"] = ticket_id

    # Verify all required parameters (those without defaults) are present
    missing_params = [
        pname
        for pname, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty and pname not in clean_args
    ]
    if missing_params:
        return {
            "error": f"Missing required argument(s) {missing_params} for '{func_name}'.",
            "success": False,
        }, False

    # Check duplicate prevention cache
    try:
        cache_key = (func_name, json.dumps(clean_args, sort_keys=True, default=str))
    except TypeError:
        cache_key = (func_name, str(clean_args))

    if cache_key in executed_cache:
        cached = dict(executed_cache[cache_key])
        cached["duplicate_call_prevented"] = True
        return cached, False

    try:
        result = target_func(**clean_args)
        if not isinstance(result, dict):
            result = {"result": result}
        executed_cache[cache_key] = result
        return result, True
    except Exception as exc:
        return {
            "error": f"Execution failed in '{func_name}': {str(exc)}",
            "success": False,
        }, False


def _build_function_response_part(call: Any, tool_result: Dict[str, Any]) -> types.Part:
    """Create a FunctionResponse Part matched to the original call's ID and name."""
    call_name = getattr(call, "name", "unknown_tool")
    call_id = getattr(call, "id", None)

    if isinstance(call_id, str) and call_id:
        return types.Part(
            function_response=types.FunctionResponse(
                id=call_id,
                name=call_name,
                response=tool_result,
            )
        )
    return types.Part.from_function_response(
        name=call_name,
        response=tool_result,
    )


# ---------------------------------------------------------------------------
# Output Parsing & Validation
# ---------------------------------------------------------------------------
def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract and parse a JSON object from model output text."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    cleaned = raw_text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        # Try extracting first {...} block
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return None
    return None


def validate_final_output(
    raw_text: Optional[str],
    ticket_id: str,
    executed_tools: List[str],
    tool_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate and normalize the final structured triage result."""
    # Determine authoritative fallback category from classify_ticket_category if executed
    classified_category: Optional[str] = None
    needs_review_from_tools = False

    for entry in tool_results:
        res = entry.get("result", {})
        if entry.get("tool") == "classify_ticket_category" and res.get("category") in VALID_CATEGORIES:
            classified_category = res["category"]
        if entry.get("tool") == "check_refund_eligibility" and res.get("requires_human_review"):
            needs_review_from_tools = True
        if entry.get("tool") == "escalate_to_human" and res.get("escalated"):
            needs_review_from_tools = True
        if res.get("found") is False and entry.get("tool") == "lookup_order_status":
            needs_review_from_tools = True

    parsed = _extract_json_object(raw_text or "")
    malformed_output = False

    if parsed is not None:
        raw_cat = str(parsed.get("category", "")).strip().lower()
        # Prefer explicit tool classification over hallucinated category
        if classified_category is not None:
            category = classified_category
        elif raw_cat in VALID_CATEGORIES:
            category = raw_cat
        else:
            category = "other"

        draft_reply = parsed.get("draft_reply")
        if not isinstance(draft_reply, str) or not draft_reply.strip():
            draft_reply = (
                "Thank you for contacting support. Your ticket has been logged and "
                "requires additional review by our support team."
            )
            malformed_output = True
    else:
        malformed_output = True
        category = classified_category or "other"
        if isinstance(raw_text, str) and raw_text.strip():
            draft_reply = raw_text.strip()
        else:
            draft_reply = (
                "We received your request, but encountered an issue generating a complete "
                "response. A support specialist will review your ticket."
            )

    return {
        "ticket_id": ticket_id,
        "category": category,
        "tools_called": list(executed_tools),
        "draft_reply": draft_reply.strip(),
        "requires_human_review": needs_review_from_tools or malformed_output,
        "malformed_output": malformed_output,
    }


# ---------------------------------------------------------------------------
# Multi-Round Ticket Triage Loop
# ---------------------------------------------------------------------------
def triage_ticket(
    ticket: Dict[str, Any],
    client: genai.Client,
    model_name: str = DEFAULT_MODEL,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the multi-round Gemini function-calling triage workflow for one ticket.

    Args:
        ticket: Dict containing ticket_id, subject, body, customer_email, order_id.
        client: Initialized google.genai.Client.
        model_name: Gemini model identifier.
        max_tool_rounds: Safety cap on consecutive tool-calling rounds (default: 5).
        verbose: Print detailed tool invocations and results if True.

    Returns:
        Structured triage report dict including ticket_id, category, tools_called,
        tool_results, draft_reply, requires_human_review, safety_cap_reached, tool_rounds.
    """
    ticket_id = str(ticket.get("ticket_id", "UNKNOWN"))
    subject = str(ticket.get("subject", ""))
    body = str(ticket.get("body", ""))
    customer_email = str(ticket.get("customer_email", ""))
    order_id = ticket.get("order_id")

    user_prompt = (
        f"Please triage the following customer support ticket:\n"
        f"Ticket ID: {ticket_id}\n"
        f"Customer Email: {customer_email}\n"
        f"Order ID: {order_id if order_id else 'None provided'}\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n"
    )

    config = types.GenerateContentConfig(
        tools=[TICKET_TOOLS],
        temperature=0.0,
        system_instruction=TICKET_SYSTEM_INSTRUCTION,
    )

    contents: List[Any] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)],
        )
    ]

    executed_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    executed_tools: List[str] = []
    tool_results: List[Dict[str, Any]] = []
    tool_rounds_executed = 0

    while True:
        try:
            response = _call_gemini_with_retry(
                client=client,
                model_name=model_name,
                contents=contents,
                config=config,
                verbose=verbose,
            )
        except errors.APIError as api_err:
            err_msg = f"Gemini API Error: {getattr(api_err, 'message', str(api_err))} (code: {getattr(api_err, 'code', 'unknown')})"
            if verbose:
                print(f"  [ERROR] {err_msg}")
            return {
                "ticket_id": ticket_id,
                "subject": subject,
                "category": "other",
                "tools_called": executed_tools,
                "tool_results": tool_results,
                "draft_reply": "",
                "requires_human_review": True,
                "safety_cap_reached": False,
                "tool_rounds": tool_rounds_executed,
                "error": err_msg,
            }
        except Exception as exc:
            err_msg = f"Unexpected error communicating with Gemini: {str(exc)}"
            if verbose:
                print(f"  [ERROR] {err_msg}")
            return {
                "ticket_id": ticket_id,
                "subject": subject,
                "category": "other",
                "tools_called": executed_tools,
                "tool_results": tool_results,
                "draft_reply": "",
                "requires_human_review": True,
                "safety_cap_reached": False,
                "tool_rounds": tool_rounds_executed,
                "error": err_msg,
            }

        function_calls = getattr(response, "function_calls", None)

        # Final answer reached when Gemini stops requesting tools
        if not function_calls:
            validated = validate_final_output(
                raw_text=getattr(response, "text", None),
                ticket_id=ticket_id,
                executed_tools=executed_tools,
                tool_results=tool_results,
            )
            return {
                "ticket_id": ticket_id,
                "subject": subject,
                "category": validated["category"],
                "tools_called": validated["tools_called"],
                "tool_results": tool_results,
                "draft_reply": validated["draft_reply"],
                "requires_human_review": validated["requires_human_review"],
                "safety_cap_reached": False,
                "tool_rounds": tool_rounds_executed,
                "error": None,
            }

        # Enforce safety cap BEFORE executing another tool round
        if tool_rounds_executed >= max_tool_rounds:
            warning_msg = (
                f"SAFETY CAP REACHED: Model continued requesting tools after "
                f"{max_tool_rounds} round(s) for ticket {ticket_id}. Stopping loop."
            )
            if verbose:
                print(f"  [WARNING] {warning_msg}")

            # Recover classification category if already executed
            fallback_cat = "other"
            for entry in tool_results:
                if entry.get("tool") == "classify_ticket_category":
                    cat = entry.get("result", {}).get("category")
                    if cat in VALID_CATEGORIES:
                        fallback_cat = cat
                        break

            return {
                "ticket_id": ticket_id,
                "subject": subject,
                "category": fallback_cat,
                "tools_called": executed_tools,
                "tool_results": tool_results,
                "draft_reply": (
                    "Automated triage reached the maximum tool-call safety limit. "
                    "This ticket has been flagged for human review."
                ),
                "requires_human_review": True,
                "safety_cap_reached": True,
                "tool_rounds": tool_rounds_executed,
                "error": warning_msg,
            }

        # Execute this round's requested tool calls (supports 1 or multiple calls)
        tool_rounds_executed += 1
        if verbose:
            print(f"\n  [ROUND {tool_rounds_executed}] Gemini requested {len(function_calls)} tool call(s):")

        # Preserve model's response turn in conversation history
        candidates = getattr(response, "candidates", None)
        if candidates and getattr(candidates[0], "content", None) is not None:
            contents.append(candidates[0].content)

        tool_response_parts: List[types.Part] = []

        for call in function_calls:
            call_name = getattr(call, "name", "unknown")
            call_args = getattr(call, "args", {})
            try:
                printable_args = dict(call_args) if call_args else {}
            except (TypeError, ValueError):
                printable_args = {"raw": str(call_args)}

            tool_result, freshly_executed = validate_and_execute_tool(
                call=call,
                executed_cache=executed_cache,
                ticket_id=ticket_id,
            )

            if freshly_executed or call_name not in executed_tools:
                if call_name in TOOL_REGISTRY and freshly_executed:
                    executed_tools.append(call_name)

            tool_results.append(
                {
                    "round": tool_rounds_executed,
                    "tool": call_name,
                    "arguments": printable_args,
                    "result": tool_result,
                    "freshly_executed": freshly_executed,
                }
            )

            if verbose:
                status_tag = "EXECUTED" if freshly_executed else "BLOCKED/CACHED"
                print(f"    -> [{status_tag}] {call_name}({json.dumps(printable_args)})")
                print(f"       Result: {json.dumps(tool_result)}")

            tool_response_parts.append(_build_function_response_part(call, tool_result))

        # Send all tool results back to Gemini under role='user'
        contents.append(
            types.Content(
                role="user",
                parts=tool_response_parts,
            )
        )
