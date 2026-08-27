"""Minimal agentic-loop harness for evaluating chat-completions endpoints.

Each task is a directory under tasks/ with:
  - task.md     : the user prompt
  - sandbox/    : files the agent will see and modify (copied to a workdir
                  before the run; the originals are never touched)

Tools available to the model: read_file, write_file, list_files, run_python.
Sandbox safety: paths are resolved within the workdir; escapes raise.

Usage:
  python harness.py --task tasks/bugfix
  python harness.py --task tasks/recovery --url http://localhost:8093/v1
  python harness.py --task tasks/feature_add --max-turns 30 --transcript-out runs/

Each run prints a transcript and (optionally) saves the JSON event log to
TRANSCRIPT_OUT/<task-name>-<timestamp>.json. Tests in the sandbox are run at
the end to record final pass/fail.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from crucible_bench.testbench import sandbox

DEFAULT_URL = os.environ.get("CRUCIBLE_ENDPOINT", "http://localhost:8093/v1")


def make_tools() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the sandbox.",
            "parameters": {"type": "object",
                "properties": {"path": {"type": "string"}}, "required": ["path"]},
        }},
        {"type": "function", "function": {
            "name": "write_file",
            "description": "Overwrite a sandbox file with new contents.",
            "parameters": {"type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"]},
        }},
        {"type": "function", "function": {
            "name": "list_files",
            "description": "List files in the sandbox.",
            "parameters": {"type": "object", "properties": {}},
        }},
        {"type": "function", "function": {
            "name": "run_python",
            "description": "Run a Python script in the sandbox; returns exit_code, stdout, stderr.",
            "parameters": {"type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path"]},
        }},
    ]


def safe_path(workdir: str, name: str) -> str:
    full = os.path.realpath(os.path.join(workdir, name))
    root = os.path.realpath(workdir)
    if not (full == root or full.startswith(root + os.sep)):
        raise ValueError(f"path escapes sandbox: {name}")
    return full


def execute_tool(workdir: str, name: str, args: dict) -> str:
    try:
        if name == "read_file":
            with open(safe_path(workdir, args["path"])) as f:
                return f.read()
        if name == "write_file":
            with open(safe_path(workdir, args["path"]), "w") as f:
                f.write(args["content"])
            return f"wrote {len(args['content'])} bytes to {args['path']}"
        if name == "list_files":
            return "\n".join(sorted(os.listdir(workdir)))
        if name == "run_python":
            # safe_path keeps the *target* inside the workspace; it says nothing
            # about what the file does once running. The model wrote it, so this
            # goes through the same isolation as every other benchmark here.
            safe_path(workdir, args["path"])          # rejects escapes, still
            code, out, err = sandbox.run_file(
                workdir, args["path"], list(args.get("args") or []), timeout=30
            )
            return json.dumps({
                "exit_code": code,
                "stdout": out[-2500:],
                "stderr": err[-2500:],
            })
        return f"unknown tool: {name}"
    except Exception as e:
        return f"ERROR: {e}"


def chat(
    url: str, messages: list[dict], tools: list[dict], model: str = "default"
) -> tuple[dict, float]:
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": 0,
        "max_tokens": 2048,
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read()), time.time() - t0


def run_task(
    task_dir: Path, url: str, max_turns: int, workdir: str,
    model: str = "default",
) -> dict:
    """Run an agent task; return a result dict."""
    task_md = (task_dir / "task.md").read_text()
    sandbox_src = task_dir / "sandbox"

    # Copy sandbox into workdir
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    shutil.copytree(sandbox_src, workdir)
    print(f"workdir: {workdir}")
    print(f"sandbox files: {sorted(os.listdir(workdir))}")

    system = (
        "You are a coding assistant. You have tools to read, write, list, and run "
        "Python files in a sandbox. When given a task, plan briefly, execute step "
        "by step, and verify with tests. Stop when the task is complete."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task_md.strip()},
    ]
    tools = make_tools()

    print("=" * 70)
    print(f"TASK ({task_dir.name}):")
    print(task_md.strip())
    print("=" * 70)

    transcript: list[dict] = []
    finish_reason = None

    for turn in range(1, max_turns + 1):
        print(f"\n--- TURN {turn} ---")
        try:
            resp, dt = chat(url, messages, tools, model)
        except Exception as e:
            print(f"chat error: {e}")
            transcript.append({"turn": turn, "error": str(e)})
            break

        msg = resp["choices"][0]["message"]
        finish_reason = resp["choices"][0].get("finish_reason")
        usage = resp.get("usage", {})
        print(f"[{dt:.1f}s, {usage.get('prompt_tokens','?')} pt, "
              f"{usage.get('completion_tokens','?')} ct, finish={finish_reason}]")

        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        turn_log = {
            "turn": turn, "dt_s": dt,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": finish_reason,
            "content": content,
            "tool_calls": [],
        }

        if content:
            print(f"CONTENT: {content[:400]}")
        if not tool_calls:
            print(">>> no tool calls — stopping loop")
            transcript.append(turn_log)
            break

        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn = tc["function"]
            try:
                tc_args = json.loads(fn["arguments"])
            except Exception:
                tc_args = {}
            print(f"TOOL: {fn['name']}({json.dumps(tc_args)[:250]})")
            result = execute_tool(workdir, fn["name"], tc_args)
            preview = result[:300] + ("..." if len(result) > 300 else "")
            print(f"  -> {preview}")
            turn_log["tool_calls"].append({
                "name": fn["name"],
                "args": tc_args,
                "result_preview": preview,
                "result_length": len(result),
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
        transcript.append(turn_log)
    else:
        print(f">>> hit max_turns={max_turns}")

    # Final state — try to run any *_test.py / test_*.py
    print("\n" + "=" * 70)
    print("FINAL STATE")
    final_test_results = {}
    for name in sorted(os.listdir(workdir)):
        if name.startswith("test_") and name.endswith(".py"):
            code, out, err = sandbox.run_file(workdir, name, timeout=60)
            final_test_results[name] = {
                "exit_code": code,
                "stdout_tail": out[-800:],
                "stderr_tail": err[-400:],
            }
            print(f"\n--- {name} (exit={code}) ---")
            print(out[-800:] if out else "(no stdout)")
            if err:
                print("stderr:", err[-400:])

    return {
        "task": task_dir.name,
        "url": url,
        "turns": len(transcript),
        "finish_reason": finish_reason,
        "transcript": transcript,
        "final_tests": final_test_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Path to a task directory")
    ap.add_argument("--url", default=DEFAULT_URL, help="OpenAI-compatible base URL (e.g. http://host:port/v1)")
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--workdir", default=None,
                    help="Sandbox copy dir (default: /tmp/agent_probe_<task>)")
    ap.add_argument("--transcript-out", default=None, help="Directory to save full JSON transcript")
    ap.add_argument("--model", default="default",
                    help="Model id sent in the request body; single-model servers "
                         "ignore it")
    ap.add_argument("--sandbox", default="auto", choices=sandbox.SANDBOX_MODES,
                    help="How to isolate code the model writes and runs. 'auto' prefers podman, "
                         "then docker, and falls back to LOCAL EXECUTION with a warning. Use "
                         "docker/podman against any endpoint you do not control; they fail rather "
                         "than fall back.")
    ap.add_argument("--sandbox-image", default="crucible-bench-agent-probe",
                    help="Image for the sandbox. Needs pytest — two of the bundled tasks' tests "
                         "import it. Build containers/agent-probe-sandbox.Dockerfile.")
    args = ap.parse_args()

    sandbox.set_sandbox(args.sandbox, args.sandbox_image)

    task_dir = Path(args.task).resolve()
    if not (task_dir / "task.md").exists():
        sys.exit(f"missing task.md in {task_dir}")
    if not (task_dir / "sandbox").is_dir():
        sys.exit(f"missing sandbox/ in {task_dir}")

    workdir = args.workdir or os.path.join(
        tempfile.gettempdir(), f"agent_probe_{task_dir.name}"
    )
    result = run_task(task_dir, args.url, args.max_turns, workdir, args.model)

    # Final summary
    all_passed = all(r["exit_code"] == 0 for r in result["final_tests"].values())
    print("\n" + "=" * 70)
    print(f"SUMMARY: {result['turns']} turns, finish={result['finish_reason']}, "
          f"tests_all_passed={all_passed}")

    if args.transcript_out:
        out_dir = Path(args.transcript_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{task_dir.name}_{ts}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"transcript saved: {out_path}")


if __name__ == "__main__":
    main()
