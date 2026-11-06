"""Completed solution for Day 1 — LLM API Foundation."""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv


load_dotenv()


PRICING_1M_TOKENS = {
    "openai/gpt-4o": {"input": 5.00, "output": 20.00},
    "openai/gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "grok-4.7": {"input": 2.00, "output": 6.00},
}

OPENAI_MODEL = "openai/gpt-4o"
OPENAI_MINI_MODEL = "openai/gpt-4o-mini"
GEMINI_MODEL = "gemini-3.8-flash"
GROK_MODEL = "grok-4.7"


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call OpenAI and return text, elapsed seconds, and token usage."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return response.choices[0].message.content or "", latency, usage


def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call Gemini and return text, elapsed seconds, and token usage."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.perf_counter() - start
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return response.text or "", latency, usage


def call_grok(
    prompt: str,
    model: str = GROK_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call xAI Grok and return text, elapsed seconds, and token usage."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
    start = time.perf_counter()
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response.output_text or "", latency, usage


def compare_models(prompt: str) -> dict:
    """Compare GPT-4o, GPT-4o Mini, and Gemini Flash on one prompt."""
    calls = {
        "gpt4o": (call_openai(prompt, model=OPENAI_MODEL), OPENAI_MODEL),
        "gpt4o_mini": (
            call_openai(prompt, model=OPENAI_MINI_MODEL),
            OPENAI_MINI_MODEL,
        ),
        "gemini_flash": (call_gemini(prompt, model=GEMINI_MODEL), GEMINI_MODEL),
    }
    result = {}
    for key, (call_result, model) in calls.items():
        response, latency, usage = call_result
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        rates = PRICING_1M_TOKENS[model]
        cost = (
            input_tokens * rates["input"] + output_tokens * rates["output"]
        ) / 1_000_000
        result[key] = {
            "response": response,
            "latency": latency,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    return result


def streaming_chatbot() -> None:
    """Run a Gemini streaming chatbot retaining the last three turns."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            return
        if not user_input:
            continue

        user_message = types.Content(
            role="user", parts=[types.Part(text=user_input)]
        )
        response_parts = []
        print("Gemini: ", end="", flush=True)
        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=history + [user_message],
        )
        for chunk in stream:
            chunk_text = chunk.text or ""
            response_parts.append(chunk_text)
            print(chunk_text, end="", flush=True)
        print()

        model_message = types.Content(
            role="model", parts=[types.Part(text="".join(response_parts))]
        )
        history.extend([user_message, model_message])
        history = history[-6:]


def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Retry a zero-argument callable using exponential backoff."""
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
            time.sleep(base_delay * (2 ** attempt))


def batch_compare(prompts: list[str]) -> list[dict]:
    """Run model comparison for every prompt."""
    return [{"prompt": prompt, **compare_models(prompt)} for prompt in prompts]


def format_comparison_table(results: list[dict]) -> str:
    """Render batch comparison results as a Markdown table."""
    rows = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---:|---:|---:|",
    ]
    model_labels = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }

    def clean_cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    def truncate(value: Any, limit: int = 50) -> str:
        text = clean_cell(value)
        return text if len(text) <= limit else text[: limit - 3] + "..."

    for result in results:
        prompt = clean_cell(result.get("prompt", ""))
        for key, label in model_labels.items():
            stats = result[key]
            rows.append(
                f"| {prompt} | {label} | {truncate(stats['response'])} | "
                f"{stats['latency']:.3f}s | "
                f"{stats['input_tokens']}/{stats['output_tokens']} | "
                f"${stats['cost']:.8f} |"
            )
    return "\n".join(rows)

