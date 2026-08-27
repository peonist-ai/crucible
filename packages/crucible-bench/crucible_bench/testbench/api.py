"""Talking to an OpenAI-compatible endpoint, and cleaning up what comes back.

Deliberately urllib and not `requests` or the openai SDK: the whole test bench
sends one JSON body and reads one JSON body, and a benchmark harness is a bad
place to acquire dependencies.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request


def chat_completion(
    url: str, message: str, max_tokens: int, temperature: float,
    extra_body: dict | None = None,
) -> tuple[str, dict]:
    """Send a chat completion request. Returns (text, meta).

    Args:
        extra_body: Merged into the request body verbatim.

    This used to translate a `thinking_budget` argument into
    `chat_template_kwargs: {enable_thinking: true}` plus
    `thinking_token_budget` -- one model family's dialect for "reason, but not
    too much", hard-coded into the client every benchmark shares.

    Newer servers spell it `reasoning_effort: minimal|low|...`, and the old
    keys are simply unknown to them. Measured 2026-08-23 against a Qwen 3.8
    seat: both keys were accepted and silently ignored, while the results file
    recorded `thinking_budget: 2048` as though it had been honoured. The
    harness was reporting a generation setting that never reached the model.

    So the client no longer guesses the dialect. Callers pass whatever their
    server actually speaks, and `_check_ignored_params` asks the server which
    of those it intends to ignore.
    """
    body: dict = {
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if extra_body:
        body.update(extra_body)

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    # Reasoning responses can be long — extend the timeout whenever any
    # generation option is in play, since that is what turns them on.
    timeout = 600 if extra_body else 120
    data, error, attempts = _post_with_retry(req, timeout)
    if data is None:
        return f"ERROR: {error}", {
            "finish_reason": "error", "error": str(error)[:200],
            "attempts": attempts,
        }

    choice = data["choices"][0]
    message = choice["message"]
    return strip_thinking(message.get("content") or ""), {
        # Why generation stopped, and how much of the budget reasoning ate.
        # A response whose `content` is empty because the model spent its
        # whole budget thinking is a budget failure, and it looks exactly
        # like a model that could not answer -- measured on GPQA at
        # reasoning_effort=low, 6 of 26 responses came back empty and were
        # scored wrong with nothing in the record to say why.
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
        "reasoning_chars": len(message.get("reasoning_content") or ""),
    }


def text_completion(
    url: str, prompt: str, max_tokens: int, temperature: float,
    stop: list[str] | None = None,
) -> str:
    """Send a raw text completion request. Used for code generation.

    The chat endpoint triggers a thinking loop on code tasks after
    compression. The completions endpoint with natural language prompts
    works correctly (verified via llama-cli).
    """
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop or [],
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["text"]
    except Exception as e:
        return f"ERROR: {e}"


# Transport failures are not model failures. A batch-1 server answering 503
# because it is busy, or dropping a connection because it was restarted
# mid-run, has told us nothing about the model -- but scoring the empty
# response as wrong silently deflates the result. Measured 2026-08-23: a server
# restart during a HumanEval+ run put 7 of 120 problems in as failures with
# `Connection reset by peer`, costing ~5.6 points that had nothing to do with
# the model.
RETRY_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 3


def _post_with_retry(req, timeout: int):
    """Send a request, retrying transport-level failures.

    Returns (data, error, attempts). Only retries things that are plausibly
    transient: connection resets, timeouts, and the 503 a single-flight server
    returns when another request is in flight. An HTTP 400 means we sent
    something wrong and retrying would just ask again more slowly.
    """
    last = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read()), None, attempt
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code not in (429, 500, 502, 503, 504):
                return None, last, attempt
        except Exception as e:
            last = e
        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return None, last, RETRY_ATTEMPTS


def api_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def strip_thinking(text: str) -> str:
    """Remove Gemma 4 thinking/channel tags from response.

    Handles both vLLM (reasoning parser strips thinking) and llama.cpp
    (thinking content appears inline between channel tags).
    """
    # Block-level: remove everything between thought channel open/close
    text = re.sub(
        r"<\|channel>\s*thought\s*\n.*?<channel\|>",
        "", text, flags=re.DOTALL,
    )
    # Residual tag cleanup
    text = re.sub(r"<\|?channel>?\s*thought\s*\n?", "", text)
    text = re.sub(r"<\|?channel\|?>", "", text)
    text = re.sub(r"^\s*thought\s*\n", "", text)
    return text.strip()


def extract_code(response: str, fallback_prefix: str = "") -> str:
    """Extract Python code from a model response."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return "\n\n".join(blocks)

    lines = response.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "import ", "from ", "class ")):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    if fallback_prefix:
        return fallback_prefix + response

    return response


def truncate(s: str | None, n: int) -> str | None:
    """Cap a field before it goes into the result JSON."""
    if s and len(s) > n:
        return s[:n] + "..."
    return s
