"""Day 01 - LLM Foundation with OpenRouter, Google Gemini, and xAI."""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv


load_dotenv()


# Model IDs can be overridden in .env without changing this file.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.7")

# Estimated standard prices per 1M tokens (USD). Override them in .env when
# selecting a different model or when a provider changes its prices.
PRICING_1M_TOKENS = {
    "openrouter": {
        "input": float(os.getenv("OPENROUTER_INPUT_PRICE", "0.15")),
        "output": float(os.getenv("OPENROUTER_OUTPUT_PRICE", "0.60")),
    },
    "gemini": {
        "input": float(os.getenv("GEMINI_INPUT_PRICE", "0.75")),
        "output": float(os.getenv("GEMINI_OUTPUT_PRICE", "3.75")),
    },
    "xai": {
        "input": float(os.getenv("XAI_INPUT_PRICE", "2.00")),
        "output": float(os.getenv("XAI_OUTPUT_PRICE", "6.00")),
    },
}


def _require_api_key(name: str) -> str:
    """Return a configured API key or raise a helpful error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Add it to the project .env file.")
    if name == "XAI_API_KEY" and value.startswith("gsk_"):
        raise RuntimeError(
            "XAI_API_KEY appears to contain a Groq key (gsk_), not an xAI key. "
            "Create an xAI key in the xAI Console and update .env."
        )
    return value


def _usage_value(usage: Any, *names: str) -> int:
    """Read the first available token counter from an SDK usage object."""
    if usage is None:
        return 0
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


# ---------------------------------------------------------------------------
# Task 1 - Call a model through OpenRouter
# ---------------------------------------------------------------------------
def call_openrouter(
    prompt: str,
    model: str = OPENROUTER_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call OpenRouter and return text, latency, and token usage."""
    from openai import OpenAI

    client = OpenAI(
        api_key=_require_api_key("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    started_at = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - started_at
    usage_data = response.usage
    usage = {
        "input_tokens": _usage_value(usage_data, "prompt_tokens", "input_tokens"),
        "output_tokens": _usage_value(
            usage_data, "completion_tokens", "output_tokens"
        ),
    }
    return response.choices[0].message.content or "", latency, usage


# ---------------------------------------------------------------------------
# Task 2 - Call Google Gemini
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call Gemini and return text, latency, and token usage."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_require_api_key("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    started_at = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.perf_counter() - started_at
    usage_data = response.usage_metadata
    usage = {
        "input_tokens": _usage_value(
            usage_data, "prompt_token_count", "input_tokens"
        ),
        "output_tokens": _usage_value(
            usage_data, "candidates_token_count", "output_tokens"
        ),
    }
    return response.text or "", latency, usage


# ---------------------------------------------------------------------------
# Task 3 - Call xAI Grok
# ---------------------------------------------------------------------------
def call_xai(
    prompt: str,
    model: str = XAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call the xAI Responses API and return text, latency, and token usage."""
    from openai import OpenAI

    client = OpenAI(
        api_key=_require_api_key("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
    started_at = time.perf_counter()
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    latency = time.perf_counter() - started_at
    usage_data = response.usage
    usage = {
        "input_tokens": _usage_value(usage_data, "input_tokens", "prompt_tokens"),
        "output_tokens": _usage_value(
            usage_data, "output_tokens", "completion_tokens"
        ),
    }
    return response.output_text or "", latency, usage


# ---------------------------------------------------------------------------
# Task 4 - Compare OpenRouter, Gemini, and xAI
# ---------------------------------------------------------------------------
def _make_comparison_result(
    call_result: tuple[str, float, dict], provider: str
) -> dict:
    response, latency, usage = call_result
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    rates = PRICING_1M_TOKENS[provider]
    cost = (
        input_tokens * rates["input"] + output_tokens * rates["output"]
    ) / 1_000_000
    return {
        "response": response,
        "latency": latency,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def compare_models(prompt: str) -> dict:
    """Send one prompt to all three providers and compare their results."""
    return {
        "openrouter": _make_comparison_result(
            call_openrouter(prompt), "openrouter"
        ),
        "gemini": _make_comparison_result(call_gemini(prompt), "gemini"),
        "xai": _make_comparison_result(call_xai(prompt), "xai"),
    }


# ---------------------------------------------------------------------------
# Task 5 - Streaming chatbot with Gemini (last three conversation turns)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """Run an interactive Gemini chatbot and retain the last three turns."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_require_api_key("GEMINI_API_KEY"))
    history: list[Any] = []

    print("\n=== Gemini Streaming Chatbot ===")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                print("Goodbye!")
                return

            user_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
            response_parts: list[str] = []
            print("Gemini: ", end="", flush=True)

            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=history + [user_message],
            )
            for chunk in stream:
                chunk_text = chunk.text or ""
                response_parts.append(chunk_text)
                print(chunk_text, end="", flush=True)
            print("\n")

            model_message = types.Content(
                role="model",
                parts=[types.Part.from_text(text="".join(response_parts))],
            )
            history.extend([user_message, model_message])
            history = history[-6:]
        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            return
        except Exception as error:
            print(f"\n[API error] {error}\n")


# ---------------------------------------------------------------------------
# Bonus Task A - Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Retry fn with delays base_delay * 2**attempt."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2**attempt))


# ---------------------------------------------------------------------------
# Bonus Task B - Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """Run compare_models for every prompt and retain per-prompt errors."""
    results = []
    for prompt in prompts:
        try:
            results.append({"prompt": prompt, **compare_models(prompt)})
        except Exception as error:
            error_result = {
                "response": f"Error: {error}",
                "latency": 0.0,
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            results.append(
                {
                    "prompt": prompt,
                    "openrouter": dict(error_result),
                    "gemini": dict(error_result),
                    "xai": dict(error_result),
                }
            )
    return results


# ---------------------------------------------------------------------------
# Bonus Task C - Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """Format batch comparison results as a Markdown table."""
    rows = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---:|---:|---:|",
    ]
    labels = {
        "openrouter": f"OpenRouter ({OPENROUTER_MODEL})",
        "gemini": f"Gemini ({GEMINI_MODEL})",
        "xai": f"xAI ({XAI_MODEL})",
    }

    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    def truncate(value: Any, limit: int = 50) -> str:
        text = clean(value)
        return text if len(text) <= limit else text[: limit - 3] + "..."

    for result in results:
        prompt = truncate(result.get("prompt", ""), 40)
        for provider, label in labels.items():
            stats = result[provider]
            rows.append(
                f"| {prompt} | {label} | {truncate(stats['response'])} | "
                f"{stats['latency']:.3f}s | "
                f"{stats['input_tokens']}/{stats['output_tokens']} | "
                f"${stats['cost']:.8f} |"
            )
    return "\n".join(rows)


if __name__ == "__main__":
    prompt = (
        "Hay giai thich su khac nhau giua temperature va top_p "
        "bang tieng Viet, ngan gon trong 2 cau."
    )
    print("=== OpenRouter vs Gemini vs xAI ===")
    try:
        comparison = compare_models(prompt)
        print(format_comparison_table([{"prompt": prompt, **comparison}]))
    except Exception as error:
        print(f"Comparison failed: {error}")

    try:
        streaming_chatbot()
    except Exception as error:
        print(f"Chatbot failed to start: {error}")
