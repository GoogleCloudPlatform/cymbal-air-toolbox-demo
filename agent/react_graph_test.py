# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver

from agent import react_graph


def test_unknown_tool_call_returns_error_message(monkeypatch):
    class FakeChatVertexAI:
        def __init__(self, **_):
            self.invocation_count = 0

        def bind_tools(self, _):
            async def respond(_):
                self.invocation_count += 1
                if self.invocation_count == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "unknown_tool",
                                "args": {},
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                return AIMessage(content="The requested tool is unavailable.")

            return RunnableLambda(respond)

    monkeypatch.setattr(react_graph, "ChatVertexAI", FakeChatVertexAI)

    async def run_graph():
        prompt = ChatPromptTemplate.from_messages([("placeholder", "{messages}")])
        graph = await react_graph.create_graph(
            tools=[],
            insert_ticket=None,
            validate_ticket=None,
            checkpointer=MemorySaver(),
            prompt=prompt,
            model_name="unused",
            debug=False,
        )
        return await graph.ainvoke(
            {"messages": [HumanMessage(content="Use an unavailable tool")]},
            config={"configurable": {"thread_id": "test-thread"}},
        )

    result = asyncio.run(run_graph())

    tool_message = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert tool_message.name == "unknown_tool"
    assert tool_message.content == "Error: Tool 'unknown_tool' not found."
