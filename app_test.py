# Copyright 2023 Google LLC
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

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from agent import Agent
from app import init_app


class TestAgentErrorHandling(unittest.IsolatedAsyncioTestCase):
    async def test_user_session_invoke_chat_error(self):
        agent = Agent()

        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {"messages": []}
        mock_graph.get_state.return_value = mock_state

        mock_graph.ainvoke = AsyncMock(side_effect=Exception("LLM API failed"))

        agent._langgraph_app = mock_graph
        agent._user_sessions["test_uuid"] = "some_token"

        response = await agent.user_session_invoke("test_uuid", "hello")

        self.assertEqual(
            response["output"], "Sorry, we couldn't answer your question 😢"
        )

        # Verify we synchronized both user input and AI error back to the graph's history
        mock_graph.update_state.assert_called_once()
        called_args, called_kwargs = mock_graph.update_state.call_args

        state_update = called_kwargs.get("values", called_args[1])
        messages = state_update["messages"]
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertEqual(messages[0].content, "hello")
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(
            messages[1].content, "Sorry, we couldn't answer your question 😢"
        )

    async def test_user_session_invoke_confirm_booking_error(self):
        agent = Agent()

        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {"messages": []}
        mock_graph.get_state.return_value = mock_state
        # Ticket insertion should fail
        mock_graph.ainvoke = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        agent._langgraph_app = mock_graph
        agent._user_sessions["test_uuid"] = "some_token"

        response = await agent.user_session_insert_ticket("test_uuid")

        self.assertEqual(response["output"], "Sorry, flight booking failed. 😢")

        # Make sure both "Book it" and the specific error reply are saved to persistent state
        mock_graph.update_state.assert_called_once()
        called_args, called_kwargs = mock_graph.update_state.call_args
        state_update = called_kwargs.get("values", called_args[1])
        messages = state_update["messages"]
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertEqual(messages[0].content, "Looks good to me. Book it!")
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(messages[1].content, "Sorry, flight booking failed. 😢")

    async def test_user_session_invoke_decline_booking_error(self):
        agent = Agent()

        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {"messages": []}
        mock_graph.get_state.return_value = mock_state
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("Cancel failed"))

        agent._langgraph_app = mock_graph
        agent._user_sessions["test_uuid"] = "some_token"

        response = await agent.user_session_decline_ticket("test_uuid")

        self.assertEqual(response["output"], "Sorry, something went wrong. 😢")

        # Note: during decline, update_state gets called twice.
        # First BEFORE invoking (for the decline intent), then inside the except block.
        self.assertEqual(mock_graph.update_state.call_count, 2)

        # Ensure the second call records the specific fallback error message
        second_call_args, second_call_kwargs = mock_graph.update_state.call_args_list[1]
        state_update = second_call_kwargs.get("values", second_call_args[1])
        messages = state_update["messages"]
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], AIMessage)
        self.assertEqual(messages[0].content, "Sorry, something went wrong. 😢")


class TestAppEndpointsErrorHandling(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = init_app(client_id="test_client_id", middleware_secret="test_secret")
        self.client = TestClient(self.app)

    @patch("agent.agent.initialize_tools", new_callable=AsyncMock)
    @patch("agent.agent.create_graph", new_callable=AsyncMock)
    async def test_chat_endpoint_error_handling(
        self, mock_create_graph, mock_initialize_tools
    ):
        mock_agent = MagicMock()
        mock_agent.user_session_exist.return_value = True
        mock_agent.get_user_id_token.return_value = "dummy_token"

        async def mock_user_session_create(session):
            session["uuid"] = "test_uuid"
            session["history"] = [{"type": "ai", "data": {"content": "Welcome"}}]

        mock_agent.user_session_create = AsyncMock(side_effect=mock_user_session_create)

        mock_agent.user_session_invoke = AsyncMock(
            return_value={
                "output": "Sorry, we couldn't answer your question 😢",
                "trace": [],
            }
        )

        self.app.state.agent = mock_agent

        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        chat_response = self.client.post("/chat", json={"prompt": "test question"})
        self.assertEqual(chat_response.status_code, 200)

        data = chat_response.json()
        self.assertEqual(data["type"], "message")
        self.assertIn("Sorry, we couldn't answer your question", data["content"])

    @patch("agent.agent.initialize_tools", new_callable=AsyncMock)
    @patch("agent.agent.create_graph", new_callable=AsyncMock)
    async def test_booking_flight_confirm_error_handling(
        self, mock_create_graph, mock_initialize_tools
    ):
        mock_agent = MagicMock()
        mock_agent.user_session_exist.return_value = True
        mock_agent.get_user_id_token.return_value = "dummy_token"

        async def mock_user_session_create(session):
            session["uuid"] = "test_uuid"
            session["history"] = [{"type": "ai", "data": {"content": "Welcome"}}]

        mock_agent.user_session_create = AsyncMock(side_effect=mock_user_session_create)

        mock_agent.user_session_insert_ticket = AsyncMock(
            return_value={"output": "Sorry, flight booking failed. 😢", "trace": []}
        )

        self.app.state.agent = mock_agent

        self.client.get("/")

        response = self.client.post("/book/flight")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Sorry, flight booking failed. 😢")

    @patch("agent.agent.initialize_tools", new_callable=AsyncMock)
    @patch("agent.agent.create_graph", new_callable=AsyncMock)
    async def test_booking_flight_decline_error_handling(
        self, mock_create_graph, mock_initialize_tools
    ):
        mock_agent = MagicMock()
        mock_agent.user_session_exist.return_value = True
        mock_agent.get_user_id_token.return_value = "dummy_token"

        async def mock_user_session_create(session):
            session["uuid"] = "test_uuid"
            session["history"] = [{"type": "ai", "data": {"content": "Welcome"}}]

        mock_agent.user_session_create = AsyncMock(side_effect=mock_user_session_create)

        mock_agent.user_session_decline_ticket = AsyncMock(
            return_value={"output": "Sorry, something went wrong. 😢", "trace": []}
        )

        self.app.state.agent = mock_agent

        self.client.get("/")

        response = self.client.post("/book/flight/decline")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Sorry, something went wrong. 😢")
