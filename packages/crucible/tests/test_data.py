"""Calibration message extraction — especially tool-calling agent traces.

Modern agent datasets put the assistant's payload in `tool_calls` and leave
`content` empty. Dropping those turns loses exactly the activations a
tool-calling specialist is being calibrated for, and breaks role alternation
into user -> tool -> tool. These tests pin that behavior.
"""

from crucible.data import (
    _extract_messages,
    _flatten_tool_calls,
    _message_text_len,
    _plain_render,
    _tool_calls_missing,
    _tool_calls_to_text,
)


def _agent_trace():
    """Shape taken from choucsan/mimo-claude-code-traces-1k."""
    return {
        "messages": [
            {"role": "user", "content": "Implement Kahn's algorithm and test it."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": {"file_path": "/repo/graph.py"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "def topo(): pass", "tool_call_id": "c1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "arguments": '{"content": "from collections import deque"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "c2"},
            {"role": "assistant", "content": "Done — added Kahn's algorithm."},
        ]
    }


class TestToolCallExtraction:
    def test_tool_call_turns_survive(self):
        messages = _extract_messages(_agent_trace(), "messages")

        assert [m["role"] for m in messages] == [
            "user", "assistant", "tool", "assistant", "tool", "assistant",
        ]
        assert messages[1]["tool_calls"], "payload must be preserved for the template"
        assert messages[3]["tool_calls"]

    def test_tool_call_ids_passed_through(self):
        messages = _extract_messages(_agent_trace(), "messages")
        assert messages[2]["tool_call_id"] == "c1"

    def test_length_filter_counts_tool_calls(self):
        messages = _extract_messages(_agent_trace(), "messages")
        content_only = sum(len(m.get("content", "")) for m in messages)

        # Counting content alone undercounts badly — that's what let
        # tool-call-heavy samples fall under the 50-char floor and get dropped.
        assert _message_text_len(messages) > content_only

    def test_empty_tool_calls_do_not_create_dangling_turns(self):
        junk = {"messages": [{"role": "assistant", "tool_calls": ["nope", {"function": None}]}]}
        assert _extract_messages(junk, "messages") is None


class TestToolCallRendering:
    def test_missing_tool_calls_are_detected(self):
        messages = _extract_messages(_agent_trace(), "messages")

        # A renderer with no tool-call support silently omits them: no
        # exception, just a string missing every call the agent made.
        naive = _plain_render(messages)
        assert _tool_calls_missing(messages, naive) is True

    def test_flattening_restores_them(self):
        messages = _extract_messages(_agent_trace(), "messages")
        flat = _plain_render(_flatten_tool_calls(messages))

        assert _tool_calls_missing(messages, flat) is False
        assert "Read" in flat
        assert "Write" in flat
        assert "graph.py" in flat

    def test_flatten_preserves_existing_content(self):
        messages = [
            {"role": "assistant", "content": "let me look",
             "tool_calls": [{"function": {"name": "Read", "arguments": {"p": 1}}}]},
        ]
        merged = _flatten_tool_calls(messages)[0]["content"]

        assert merged.startswith("let me look")
        assert "Read" in merged

    def test_rendered_form_is_model_neutral(self):
        # This path only runs for templates with no native tool-call syntax,
        # so it must not inject another model's markup.
        text = _tool_calls_to_text([{"function": {"name": "ls", "arguments": {}}}])

        assert "<tool_call>" not in text
        assert '"name": "ls"' in text

    def test_string_arguments_are_not_double_encoded(self):
        text = _tool_calls_to_text(
            [{"function": {"name": "Write", "arguments": '{"path": "a.py"}'}}]
        )
        assert '"arguments": {"path": "a.py"}' in text


class TestExistingFormatsUnaffected:
    def test_sharegpt_still_works(self):
        sharegpt = {
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "list the files"},
                {"from": "gpt", "value": '<tool_call>{"name": "ls"}</tool_call>'},
                {"from": "tool", "value": "a.py b.py"},
            ]
        }
        messages = _extract_messages(sharegpt, "conversations")

        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
        # Hermes-style traces carry the call inline in `value`, so nothing to attach.
        assert not any("tool_calls" in m for m in messages)

    def test_plain_text_field(self):
        assert _extract_messages({"text": "hello world"}, "text") == [
            {"role": "user", "content": "hello world"}
        ]

    def test_json_string_messages(self):
        example = {"messages_json": '[{"role": "user", "content": "hi"}]'}
        assert _extract_messages(example, "messages_json") == [
            {"role": "user", "content": "hi"}
        ]

    def test_multipart_content_is_flattened(self):
        example = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "describe this"}]}
            ]
        }
        assert _extract_messages(example, "messages")[0]["content"] == "describe this"
