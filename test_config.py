import os

import pytest
import yaml


def load_yaml(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def verify_config_structure(config):
    assert "sources" in config, "Missing 'sources' section"
    assert "tools" in config, "Missing 'tools' section"
    assert "toolsets" in config, "Missing 'toolsets' section"

    # Verify sources
    assert len(config["sources"]) > 0, "No sources defined"
    for source_name, source_info in config["sources"].items():
        assert "kind" in source_info, f"Source '{source_name}' missing 'kind'"

    # Verify tools
    assert len(config["tools"]) > 0, "No tools defined"
    for tool_name, tool_info in config["tools"].items():
        assert "kind" in tool_info, f"Tool '{tool_name}' missing 'kind'"
        assert "source" in tool_info, f"Tool '{tool_name}' missing 'source'"
        assert "statement" in tool_info, f"Tool '{tool_name}' missing 'statement'"
        assert (
            tool_info["source"] in config["sources"]
        ), f"Tool '{tool_name}' references undefined source '{tool_info['source']}'"


def test_tools_yaml():
    config = load_yaml("tools.yaml")
    verify_config_structure(config)

    # Specific checks for Postgres/AlloyDB config
    for tool_name, tool_info in config["tools"].items():
        if tool_info["kind"] == "postgres-sql":
            # Check for Postgres specific syntax
            # This is a bit heuristic, but helps
            statement = tool_info["statement"]
            if "ILIKE" in statement:
                # ILIKE is Postgres specific
                pass


def test_tools_mysql_yaml():
    config = load_yaml("tools_mysql.yaml")
    verify_config_structure(config)

    # Specific checks for MySQL config
    for tool_name, tool_info in config["tools"].items():
        assert (
            tool_info["kind"] == "mysql-sql"
        ), f"Tool '{tool_name}' should be 'mysql-sql' in tools_mysql.yaml"

        statement = tool_info["statement"]
        assert (
            "ILIKE" not in statement
        ), f"Tool '{tool_name}' uses ILIKE which is not standard in MySQL"

        if "approx_distance" in statement:
            assert (
                "mysql.ml_embedding" in statement
            ), f"Tool '{tool_name}' uses approx_distance without mysql.ml_embedding"
