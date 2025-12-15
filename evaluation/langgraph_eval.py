# Copyright 2024 Google LLC
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
from vertexai.evaluation import EvalTask, _base as evaluation_base
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_core.agents import AgentActionMessageLog

from eval_data_models import MyEvalData, MyToolCall, RetrievalAppState
from eval_metrics import response_phase_metrics, retrieval_phase_metrics

class LangGraphEvalRunner:
    def __init__(self, agent_graph: Any): 
        self.agent_graph = agent_graph

    async def run_eval_for_query(self, eval_data: MyEvalData) -> MyEvalData:
        """Runs a single query through the LangGraph agent and populates eval_data."""
        try:
            # Invoke the LangGraph agent
            initial_messages = [BaseMessage(content=eval_data.query, type="human")]
            final_state: RetrievalAppState = await self.agent_graph.ainvoke({"messages": initial_messages})

            messages = final_state.get("messages", [])
            if not messages:
                print(f"Warning: No messages in state for query: {eval_data.query}")
                return eval_data

            llm_tool_calls: List[MyToolCall] = []
            contexts: List[Dict[str, Any] | List[Dict[str, Any]]] = []
            llm_output = ""

            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            llm_tool_calls.append(MyToolCall(name=tc["name"], arguments=tc["args"]))
                    if msg.content:
                         llm_output = msg.content
                elif isinstance(msg, ToolMessage):
                    
                    pass

            eval_data.llm_tool_calls = llm_tool_calls
            eval_data.llm_output = llm_output
            eval_data.context = contexts
            eval_data.prompt = eval_data.query 
            eval_data.instruction = f"Answer user query based on context given. User query is {eval_data.query}."

        except Exception as e:
            print(f"Error invoking LangGraph agent for query '{eval_data.query}': {e}")
        return eval_data

    async def run_llm_for_eval(self, eval_list: List[MyEvalData]) -> List[MyEvalData]:
        """Generate llm_tool_calls and llm_output for golden dataset queries."""
        # Run each eval data through the agent
        updated_eval_list = await asyncio.gather(
            *[self.run_eval_for_query(data) for data in eval_list]
        )
        return updated_eval_list

    def evaluate_retrieval_phase(self, eval_datas: List[MyEvalData], experiment_name: str) -> evaluation_base.EvalResult:
        """Evaluate tool selection and arguments (retrieval phase)."""
        responses = []
        references = []
        for e in eval_datas:
            references.append(json.dumps({"tool_calls": [t.model_dump() for t in e.tool_calls]}))
            responses.append(json.dumps({"tool_calls": [t.model_dump() for t in e.llm_tool_calls]}))

        eval_dataset = pd.DataFrame({"response": responses, "reference": references})
        eval_result = EvalTask(
            dataset=eval_dataset,
            metrics=retrieval_phase_metrics,
            experiment=experiment_name,
        ).evaluate()
        return eval_result

    def evaluate_response_phase(self, eval_datas: List[MyEvalData], experiment_name: str) -> evaluation_base.EvalResult:
        """Evaluate the final response based on context (response phase)."""
        instructions = [e.instruction for e in eval_datas]
        prompts = [e.prompt for e in eval_datas]
        responses = [e.llm_output or "" for e in eval_datas]
        contexts = [
            ", ".join([json.dumps(c) for c in e.context]) if e.context else "no data retrieved"
            for e in eval_datas
        ]

        eval_dataset = pd.DataFrame({
            "instruction": instructions,
            "prompt": prompts,
            "context": contexts,
            "response": responses,
        })
        eval_result = EvalTask(
            dataset=eval_dataset,
            metrics=response_phase_metrics,
            experiment=experiment_name,
        ).evaluate()
        return eval_result

def get_langgraph_agent():
  """
  Loads and compiles the Cymbal Air LangGraph application defined in agent/react_graph.py.
  """
  print("Initializing LangGraph Agent...")
  try:
      all_tools = get_all_tools()
      insert_ticket = get_insert_ticket_tool()
      validate_ticket = get_validate_ticket_tool()

      checkpointer = MemorySaver()

      model_name = "gemini-2.5-pro"

      compiled_graph = create_graph(
          tools=all_tools,
          insert_ticket=insert_ticket,
          validate_ticket=validate_ticket,
          checkpointer=checkpointer,
          prompt=PROMPT,
          model_name=model_name,
          debug=True 
      )
      print("LangGraph Agent successfully initialized.")
      return compiled_graph
  except Exception as e:
      print(f"Error loading or compiling LangGraph agent: {e}")
      raise RuntimeError(f"Failed to initialize LangGraph agent: {e}")
