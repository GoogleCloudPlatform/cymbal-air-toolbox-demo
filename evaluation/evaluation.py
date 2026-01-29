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

import json
from typing import Any, Dict, List

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from vertexai.evaluation import EvalTask
from vertexai.evaluation import _base as evaluation_base

from .eval_golden import EvalData, ToolCall
from .metrics import response_phase_metrics, retrieval_phase_metrics

async def _run_llm_for_eval_langgraph(
    eval_list: List[EvalData], orc: Any, session: Dict, session_id: str
) -> List[EvalData]:
    """
    Populate EvalData with LangGraph orchestration results.
    `orc` is expected to be the main orchestrator object.
    """
    print(" Running evaluation using LangGraph logic...")
    for i, eval_data in enumerate(eval_list):
        print(f"\n---  LangGraph - Processing EvalData {i+1}/{len(eval_list)} ---")
        print(f" LangGraph - Query: '{eval_data.query}'")
        query_response = None
        try:
            config = orc.get_config(session_id)
            initial_messages = orc._langgraph_app.get_state(config).values.get("messages", [])
            initial_message_count = len(initial_messages)
            print(f" LangGraph - Initial message count: {initial_message_count}")

            print(f" LangGraph - Calling orc.user_session_invoke('{session_id}', '{eval_data.query}')")
            query_response = await orc.user_session_invoke(session_id, eval_data.query)
            print(f" LangGraph - orc.user_session_invoke returned for initial query.")

            if query_response and query_response.get("confirmation"):
                print(f" LangGraph - Confirmation key detected. Calling orc.user_session_insert_ticket('{session_id}').")
                if hasattr(orc, "user_session_insert_ticket"):
                    confirmation_response = await orc.user_session_insert_ticket(session_id)
                    print(f" LangGraph - user_session_insert_ticket (confirmation) returned.")
                    query_response = confirmation_response
                else:
                    print("ERROR: LangGraph - Orchestrator has no 'user_session_insert_ticket' method!")

        except Exception as e:
            print(f"ERROR: LangGraph - Error invoking agent for query '{eval_data.query}': {e}", exc_info=True)
            continue
        else:
            eval_data.llm_output = query_response.get("output")
            print(f" LangGraph - Extracted llm_output: {eval_data.llm_output}")
            llm_tool_calls = []
            contexts = []

            state = query_response.get("state", {})
            messages = state.get("messages", [])
            print(f" LangGraph - Found {len(messages)} messages in state.")

            new_messages = messages[initial_message_count:]
            print(f" LangGraph - Processing {len(new_messages)} new messages.")
            has_tool_message = False
            for j, message in enumerate(new_messages):
                print(f" LangGraph - Inspecting new message {j}: Type={type(message).__name__}")
                if isinstance(message, AIMessage) and message.tool_calls:
                    print(f"   - AIMessage has {len(message.tool_calls)} tool calls.")
                    for tc in message.tool_calls:
                        tc_dict = tc if isinstance(tc, dict) else tc.dict()
                        tool_call = ToolCall(name=tc_dict.get("name"), arguments=tc_dict.get("args"))
                        llm_tool_calls.append(tool_call)
                        print(f"     - Extracted ToolCall: name='{tool_call.name}', args='{tool_call.arguments}'")
                elif isinstance(message, ToolMessage):
                    has_tool_message = True
                    print(f"   - ToolMessage content: {message.content[:100]}...")
                    try:
                        context_data = json.loads(message.content)
                        contexts.append(context_data)
                        print(f"   - Appended JSON context: {json.dumps(context_data)[:100]}...")
                    except (json.JSONDecodeError, TypeError) as e:
                        contexts.append(message.content)
                        print(f"   - Appended string context (JSON decode error: {e}): {message.content[:100]}...")

            eval_data.llm_tool_calls = llm_tool_calls
            eval_data.context = contexts
            print(f" LangGraph - Total tool calls extracted: {len(eval_data.llm_tool_calls)}")
            print(f" LangGraph - Total contexts extracted: {len(eval_data.context)}")

            # --- Conditional Prompt/Instruction for Rater ---
            eval_data.prompt = eval_data.query
            if has_tool_message and eval_data.context:
                context_str = ", ".join([json.dumps(c) if isinstance(c, dict) else str(c) for c in eval_data.context])
                eval_data.instruction = (
                    f"Answer the user's query '{eval_data.query}' based *only* on the provided JSON tool outputs. "
                    f"The tool outputs are: {context_str}"
                )
                print(" LangGraph - Setting instruction for tool-use query with explicit context.")
            else:
                # If no tools were used, instruct the rater to use the system prompt as grounding.
                eval_data.instruction = (
                    "Answer the user's query based on the following system information: \n\n" + PROMPT
                )
                # Ensure context is empty if no tool outputs
                eval_data.context = []
                print(" LangGraph - Setting instruction for non-tool-use query (using system prompt).")

        if eval_data.reset:
            print(f" LangGraph - eval_data.reset is True. Attempting to reset session '{session_id}'.")
            if hasattr(orc, "user_session_reset"):
                orc.user_session_reset(session, session_id)
                print(f" LangGraph - Session '{session_id}' reset.")
            else:
                print("Warning: Orchestrator missing 'user_session_reset' method.")
    print(" LangGraph evaluation run complete.")
    return eval_list

async def run_llm_for_eval(
    eval_list: List[EvalData],
    orc: Any,
    session: Dict,
    session_id: str,
    orchestration_type: str,
) -> List[EvalData]:
    """
    Generate llm_tool_calls and llm_output for golden dataset query
    based on the orchestration type.
    """
    if orchestration_type == "langgraph":
        # 'orc' is the LangGraph orchestrator object
        return await _run_llm_for_eval_langgraph(eval_list, orc, session, session_id)
    else:
        raise ValueError(f"Unknown or unsupported orchestration_type: {orchestration_type}. Only 'langgraph' is supported.")


def evaluate_retrieval_phase(
    eval_datas: List[EvalData], experiment_name: str
) -> evaluation_base.EvalResult:
    """
    Run evaluation for the ability of a model to select the right tool and arguments (retrieval phase).
    """
    # Prepare evaluation task input
    responses = []
    references = []
    for e in eval_datas:
        references.append(
            json.dumps(
                {
                    "content": e.content,
                    "tool_calls": [t.model_dump() for t in e.tool_calls],
                }
            )
        )
        responses.append(
            json.dumps(
                {
                    "content": e.content,
                    "tool_calls": [t.model_dump() for t in e.llm_tool_calls],
                }
            )
        )
    eval_dataset = pd.DataFrame(
        {
            "response": responses,
            "reference": references,
        }
    )
    # Run evaluation
    eval_result = EvalTask(
        dataset=eval_dataset,
        metrics=retrieval_phase_metrics,
        experiment=experiment_name,
    ).evaluate()
    return eval_result


def evaluate_response_phase(
    eval_datas: List[EvalData], experiment_name: str
) -> evaluation_base.EvalResult:

    instructions = []
    contexts = []
    responses = []
    prompts = []

    for e in eval_datas:
        instructions.append(e.instruction)
        prompts.append(e.prompt)
        responses.append(e.llm_output or "")

        if e.context:
            context_str_list = []
            for c in e.context:
                try:
                    context_str_list.append(json.dumps(c))
                except TypeError:
                    context_str_list.append(str(c))
            contexts.append(", ".join(context_str_list))
        else:
            contexts.append(PROMPT)

    eval_dataset = pd.DataFrame(
        {
            "instruction": instructions,
            "prompt": prompts,
            "context": contexts,
            "response": responses,
        }
    )
    # Run evaluation
    eval_result = EvalTask(
        dataset=eval_dataset,
        metrics=response_phase_metrics,
        experiment=experiment_name,
    ).evaluate()
    return eval_result

PROMPT = """The Cymbal Air Customer Service Assistant helps customers of Cymbal Air with their travel needs.Cymbal Air (airline unique two letter identifier as CY) is a passenger airline offering convenient flights to many cities around the world from itshub in San Francisco.
Cymbal Air takes pride in using the latest technology to offer the best customerservice!
Cymbal Air Customer Service Assistant (or just "Assistant" for short) is designed to assist with a wide range of tasks, from answering simple questions to complex multi-query questions that require passing results from one query to another.
Using the latest AI models, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
The assistant should not answer questions about other peoples information for privacy reasons. Assistant is a powerful tool that can help answer a wide range of questions pertaining to travel on Cymbal Air as well as ammenities of San Francisco Airport.
Answer user query based on context or information given. Use tools if necessary. Respond directly if appropriate.
"""
