"""
main.py - OrderPilot: AI Order-Support Agent with Tool Calling

This module implements:
1. Secure API key loading using python-dotenv.
2. Configuration of the official Google Gen AI SDK client.
3. The complete manual tool-calling loop with Gemini function calling.
4. Robust error handling for unknown tools, invalid parameters, rate limits (429), and API issues.
5. An interactive terminal UI displaying AI analysis, tool calls, and results.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from tools import ORDER_TOOL, TOOL_REGISTRY


# ---------------------------------------------------------------------------
# 1. Configuration and Client Initialization
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.5-flash"

SYSTEM_INSTRUCTION = (
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


def load_gemini_client() -> tuple[genai.Client, str]:
    """Load configuration from environment and initialize the Google Gen AI client.

    Returns:
        tuple[genai.Client, str]: Initialized client and the selected model name.

    Raises:
        ValueError: If GEMINI_API_KEY is not found or empty.
    """
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "ERROR: GEMINI_API_KEY not found in environment or .env file.\n"
            "Please create a .env file and set GEMINI_API_KEY=your_key.\n"
            "See .env.example for guidance."
        )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()

    # Note: Function calling requires the v1beta API version in google-genai
    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1beta"}
    )
    return client, model_name


# ---------------------------------------------------------------------------
# 2. Tool Execution Logic
# ---------------------------------------------------------------------------
def execute_tool_call(call: types.FunctionCall) -> Dict[str, Any]:
    """Match and execute a model-requested function against the explicit tool registry.

    Args:
        call: The FunctionCall object containing name and arguments.

    Returns:
        Dict[str, Any]: Structured execution output from the Python function.
    """
    func_name = call.name
    raw_args = call.args or {}

    # Guard against arbitrary execution: only call registered functions
    if func_name not in TOOL_REGISTRY:
        return {
            "found": False,
            "error": f"Unknown tool '{func_name}'. Not in registered tools."
        }

    target_func = TOOL_REGISTRY[func_name]

    try:
        # Convert args mapping to a standard python dict
        clean_args = {k: v for k, v in raw_args.items()}
        result = target_func(**clean_args)
        return result
    except TypeError as te:
        return {
            "found": False,
            "error": f"Invalid arguments supplied for '{func_name}': {str(te)}"
        }
    except Exception as exc:
        return {
            "found": False,
            "error": f"Unexpected error during execution of '{func_name}': {str(exc)}"
        }


# ---------------------------------------------------------------------------
# 3. The Complete Tool-Calling Loop
# ---------------------------------------------------------------------------
def _call_gemini_with_retry(
    client: genai.Client,
    model_name: str,
    contents: List[Any],
    config: types.GenerateContentConfig,
    max_retries: int = 3,
    verbose: bool = True,
) -> Any:
    """Invoke generate_content with automatic retry on 429 rate limit errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except errors.APIError as api_err:
            if api_err.code == 429 and attempt < max_retries:
                wait_time = 20 * attempt
                if verbose:
                    print(f"\n[RATE LIMIT] Rate limit reached. Waiting {wait_time}s before retry {attempt}/{max_retries}...")
                time.sleep(wait_time)
                continue
            raise


def process_user_query(
    user_query: str,
    client: genai.Client,
    model_name: str,
    max_iterations: int = 5,
    verbose: bool = True,
) -> str:
    """Run the complete Gemini function-calling loop for a given user query.

    Workflow:
    1. Send user message with registered tool schemas.
    2. Inspect model response for function_calls.
    3. If function calls are detected, execute them from TOOL_REGISTRY.
    4. Send function responses back to Gemini.
    5. Repeat until Gemini returns a final natural-language response.

    Args:
        user_query: The customer's message.
        client: The configured GenAI client.
        model_name: The Gemini model name.
        max_iterations: Maximum loop iterations to prevent infinite loops.
        verbose: If True, prints status and tool-execution details to terminal.

    Returns:
        str: Final natural-language response from Gemini.
    """
    if verbose:
        print("\n[AI] Analyzing your request...")

    # Configure tool calling and system prompt
    config = types.GenerateContentConfig(
        tools=[ORDER_TOOL],
        temperature=0.0,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    # Conversation history contents
    contents: List[Any] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )
    ]

    for iteration in range(1, max_iterations + 1):
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

        # If the model produced no function calls, we have the final text answer
        if not response.function_calls:
            final_text = response.text or "I was unable to generate a response. Please try again."
            return final_text.strip()

        # Append model's response candidate to the ongoing conversation contents
        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        # Process each requested function call
        tool_response_parts: List[types.Part] = []

        for call in response.function_calls:
            call_args = dict(call.args) if call.args else {}

            if verbose:
                print("\n[TOOL CALL]")
                print(f"Function:  {call.name}")
                print(f"Arguments: {json.dumps(call_args, indent=2)}")

            # Execute the function
            tool_result = execute_tool_call(call)

            if verbose:
                print("\n[TOOL EXECUTION]")
                print(f"Result:    {json.dumps(tool_result, indent=2)}")

            # Package the result using official SDK Part.from_function_response
            tool_part = types.Part.from_function_response(
                name=call.name,
                response=tool_result,
            )
            tool_response_parts.append(tool_part)

        # Send tool response turn back to Gemini using role='user'
        tool_content = types.Content(
            role="user",
            parts=tool_response_parts,
        )
        contents.append(tool_content)

    return "Reached maximum tool-call iterations without a final response."


# ---------------------------------------------------------------------------
# 4. Interactive Terminal CLI
# ---------------------------------------------------------------------------
def run_interactive_loop(client: genai.Client, model_name: str) -> None:
    """Run an interactive session for the user in the terminal."""
    print("=" * 65)
    print("        ORDERPILOT — AI ORDER SUPPORT AGENT")
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


def main() -> None:
    """Entry point for OrderPilot."""
    try:
        client, model_name = load_gemini_client()
    except ValueError as err:
        print(f"\n[CONFIGURATION ERROR]\n{err}\n")
        sys.exit(1)

    # Allow passing a single question via CLI arguments for quick testing:
    # e.g.: python main.py "What is the status of order A123?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Question: {query}")
        final_answer = process_user_query(
            user_query=query,
            client=client,
            model_name=model_name,
            verbose=True,
        )
        print("\n[FINAL ANSWER]")
        print(final_answer)
    else:
        run_interactive_loop(client, model_name)


if __name__ == "__main__":
    main()
