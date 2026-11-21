# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

import pytest

from agent.react_graph import (
    EMPTY_TOOL_RESULT_MESSAGE,
    _build_tool_message,
    _normalize_tool_output,
)


class _FakeTool:
    """Minimal stand-in for a ToolboxTool."""

    def __init__(self, name: str, result=None, error: Exception | None = None):
        self.name = name
        self._result = result
        self._error = error

    async def ainvoke(self, args):
        if self._error is not None:
            raise self._error
        return self._result


def _tool_call(name: str, args=None, call_id: str = "call-1"):
    return {"name": name, "args": args or {}, "id": call_id}


@pytest.mark.parametrize(
    "empty_output", [None, "", "   ", "null", "NULL", "None", "[]", "{}", [], {}, ()]
)
def test_normalize_tool_output_replaces_empty_results(empty_output):
    assert _normalize_tool_output(empty_output) == EMPTY_TOOL_RESULT_MESSAGE


@pytest.mark.parametrize(
    "payload",
    [
        '[{"flight_number": "657"}]',
        [{"flight_number": "657"}],
        {"flight_number": "657"},
        "0",
        0,
        False,
        "No flights to Denver today",
    ],
)
def test_normalize_tool_output_keeps_non_empty_results(payload):
    assert _normalize_tool_output(payload) == payload


def test_empty_tool_result_is_reported_as_no_results():
    # Regression test for #642: `list_tickets` returning no rows used to reach
    # the model as "null", which it read as a broken tool.
    tool = _FakeTool("list_tickets", result="null")

    message = asyncio.run(_build_tool_message(_tool_call("list_tickets"), [tool], {}))

    assert message.content == EMPTY_TOOL_RESULT_MESSAGE
    assert message.name == "list_tickets"
    assert message.tool_call_id == "call-1"


def test_non_empty_tool_result_is_passed_through():
    payload = '[{"airline": "F9", "flight_number": "657"}]'
    tool = _FakeTool("list_flights", result=payload)

    message = asyncio.run(
        _build_tool_message(
            _tool_call("list_flights", {"departure_airport": "DEN"}), [tool], {}
        )
    )

    assert message.content == payload


def test_unknown_tool_returns_error_message():
    # A hallucinated tool name must not raise UnboundLocalError.
    message = asyncio.run(_build_tool_message(_tool_call("book_rocket"), [], {}))

    assert message.content == "Error: Tool 'book_rocket' not found."
    assert message.name == "book_rocket"


def test_tool_exception_returns_error_message():
    tool = _FakeTool("list_flights", error=RuntimeError("connection refused"))

    message = asyncio.run(_build_tool_message(_tool_call("list_flights"), [tool], {}))

    assert "Error executing tool list_flights" in message.content
    assert "connection refused" in message.content
