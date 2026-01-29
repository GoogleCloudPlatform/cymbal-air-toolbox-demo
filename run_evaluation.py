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
import os
import uuid

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.id_token import fetch_id_token

# Import the evaluation functions and goldens from the package
from evaluation import (
    evaluate_response_phase,
    evaluate_retrieval_phase,
    goldens,
    run_llm_for_eval,
)

try:
    from agent import Agent as LangGraphAgent
    print("Successfully imported LangGraph 'Agent' from agent package.")
except ImportError as e:
    print(f"Error: Failed to import 'Agent' from agent package: {e}")
    print("Please ensure 'agent/__init__.py' and 'agent/agent.py' exist.")
    LangGraphAgent = None


def export_metrics_table_csv(retrieval: pd.DataFrame, response: pd.DataFrame):
    """
    Export detailed metrics table to csv file
    """
    retrieval.to_csv("retrieval_eval.csv")
    response.to_csv("response_eval.csv")


def fetch_user_id_token(client_id: str):
    request = Request()
    user_id_token = fetch_id_token(request, client_id)
    return user_id_token


async def main():
    # --- Configuration ---
    USER_ID_TOKEN = os.getenv("USER_ID_TOKEN", default=None)
    CLIENT_ID = os.getenv("CLIENT_ID", default="")
    EXPORT_CSV = bool(os.getenv("EXPORT_CSV", default=False))

    RETRIEVAL_EXPERIMENT_NAME = os.getenv(
        "RETRIEVAL_EXPERIMENT_NAME", default="retrieval-phase-eval"
    )
    RESPONSE_EXPERIMENT_NAME = os.getenv(
        "RESPONSE_EXPERIMENT_NAME", default="response-phase-eval"
    )

    ORCHESTRATION_TYPE = "langgraph"
    print(f"Running evaluation for ORCHESTRATION_TYPE: {ORCHESTRATION_TYPE}")

    # --- Setup Orchestrator ---
    orc_object = None
    if LangGraphAgent is None:
        raise ImportError(
            "Cannot run 'langgraph' eval: 'Agent' class not found. "
            "Import from 'agent' package failed."
        )
    # Instantiate the LangGraph agent class
    print("Instantiating LangGraphAgent...")
    orc_object = LangGraphAgent()
    print("LangGraphAgent instantiated.")

    # --- Setup Session ---
    session_id = str(uuid.uuid4())
    session = {"uuid": session_id}

    # Create session
    if hasattr(orc_object, "user_session_create"):
        await orc_object.user_session_create(session)
    else:
        print("Warning: Orchestrator has no 'user_session_create' method. Skipping.")

    # Set auth token
    if CLIENT_ID or USER_ID_TOKEN:
        if USER_ID_TOKEN:
            user_id_token = USER_ID_TOKEN
        else:
            if not CLIENT_ID:
                raise ValueError("CLIENT_ID must be set as an environment variable.")
            print(f"Fetching ID token for CLIENT_ID: {CLIENT_ID}")
            user_id_token = fetch_user_id_token(CLIENT_ID)

        if hasattr(orc_object, "set_user_session_header"):
            orc_object.set_user_session_header(session_id, user_id_token)
        else:
            print(
                "Warning: Orchestrator has no 'set_user_session_header' method. Skipping auth."
            )
    else:
        print("Warning: CLIENT_ID or USER_ID_TOKEN not set. Auth-required tools may fail.")


    # --- Run Evaluation ---
    print("Running LLM for evaluation...")
    eval_lists = await run_llm_for_eval(
        goldens, orc_object, session, session_id, ORCHESTRATION_TYPE
    )

    print("Evaluating retrieval phase...")
    retrieval_eval_results = evaluate_retrieval_phase(
        eval_lists, RETRIEVAL_EXPERIMENT_NAME
    )

    print("Evaluating response phase...")
    response_eval_results = evaluate_response_phase(
        eval_lists, RESPONSE_EXPERIMENT_NAME
    )

    print("\n--- Evaluation Results ---")
    print(f"Retrieval phase eval results: {retrieval_eval_results.summary_metrics}")
    print(f"Response phase eval results: {response_eval_results.summary_metrics}")
    print("--------------------------\n")

    if EXPORT_CSV:
        print("Exporting metrics to CSV...")
        export_metrics_table_csv(
            retrieval_eval_results.metrics_table,
            response_eval_results.metrics_table,
        )
        print("Export complete.")

if __name__ == "__main__":
    asyncio.run(main())