import sys
from unittest.mock import MagicMock

# Mock modules that cause circular imports or environment issues
sys.modules["agent"] = MagicMock()
mock_agent_tools = MagicMock()
mock_agent_tools.TOOLBOX_URL = "http://fake-url"
sys.modules["agent.tools"] = mock_agent_tools
sys.modules["toolbox_core"] = MagicMock()

from datetime import datetime, time
from unittest.mock import AsyncMock, patch

import pytest

# Import the module under test
from data.run_database_init import __escape_sql, initialize_data


def test_escape_sql_none():
    assert __escape_sql(None) == "NULL"


def test_escape_sql_str():
    assert __escape_sql("hello") == "'hello'"
    assert __escape_sql("O'Connor") == "'O''Connor'"


def test_escape_sql_list():
    assert __escape_sql([1, 2, 3]) == "'[1, 2, 3]'"


def test_escape_sql_time():
    t = time(10, 30, 0)
    assert __escape_sql(t) == f"'{t}'"


def test_escape_sql_datetime():
    dt = datetime(2025, 1, 1, 12, 0, 0)
    assert __escape_sql(dt) == f"'{dt}'"


def test_escape_sql_int():
    assert __escape_sql(42) == 42


import asyncio


def test_initialize_data_postgres():
    async def run():
        # Mock ToolboxClient and execute_sql
        with patch("data.run_database_init.ToolboxClient") as MockToolboxClient:
            mock_toolbox = AsyncMock()
            MockToolboxClient.return_value.__aenter__.return_value = mock_toolbox

            mock_execute_sql = AsyncMock()

            async def custom_execute_sql(sql, *args, **kwargs):
                if "VERSION()" in sql:
                    return "PostgreSQL 15.4"
                return None

            mock_execute_sql.side_effect = custom_execute_sql

            mock_toolbox.load_tool = AsyncMock(return_value=mock_execute_sql)

            # Call the function with empty lists for simplicity
            await initialize_data([], [], [], [])

            # Verify execute_sql was called with Postgres specific syntax
            mock_execute_sql.assert_any_call("SELECT VERSION()")
            mock_execute_sql.assert_any_call("DROP TABLE IF EXISTS airports CASCADE")
            mock_execute_sql.assert_any_call(
                "CREATE EXTENSION IF NOT EXISTS google_ml_integration CASCADE"
            )
            mock_execute_sql.assert_any_call("CREATE EXTENSION IF NOT EXISTS vector")

    asyncio.run(run())


def test_initialize_data_mysql():
    async def run():
        # Mock ToolboxClient and execute_sql
        with patch("data.run_database_init.ToolboxClient") as MockToolboxClient:
            mock_toolbox = AsyncMock()
            MockToolboxClient.return_value.__aenter__.return_value = mock_toolbox

            mock_execute_sql = AsyncMock()

            async def custom_execute_sql(sql, *args, **kwargs):
                if "VERSION()" in sql:
                    return "8.0.33-google (MySQL Community Server)"
                return None

            mock_execute_sql.side_effect = custom_execute_sql

            mock_toolbox.load_tool = AsyncMock(return_value=mock_execute_sql)

            # Call the function with empty lists for simplicity
            await initialize_data([], [], [], [])

            # Verify execute_sql was called with MySQL specific syntax
            mock_execute_sql.assert_any_call("SELECT VERSION()")
            mock_execute_sql.assert_any_call("DROP TABLE IF EXISTS airports")

            # Verify Extensions were NOT created
            with pytest.raises(AssertionError):
                mock_execute_sql.assert_any_call(
                    "CREATE EXTENSION IF NOT EXISTS google_ml_integration CASCADE"
                )

            # Verify embedding type uses VARBINARY
            amenities_call = None
            for call in mock_execute_sql.call_args_list:
                if "CREATE TABLE amenities" in call[0][0]:
                    amenities_call = call[0][0]
                    break

            assert amenities_call is not None
            assert "USING VARBINARY" in amenities_call

    asyncio.run(run())
