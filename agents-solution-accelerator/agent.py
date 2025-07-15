from typing import Any, Generator, Optional, Sequence, Union
import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, Tool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from pydantic import BaseModel, Field

# testing
from langchain_openai import ChatOpenAI

mlflow.langchain.autolog()

client = DatabricksFunctionClient()
set_uc_function_client(client)

############################################
# Define your Schema definition for returning structured output
############################################
class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""

    answer: str = Field(
        description="The answer to the user's question without doctor's location data"
    )
    first_doctor: str = Field(description="The first recommended doctor's name")
    first_doctor_lattitude: str = Field(
        description="The first recommended doctor's latitude and direction"
    )
    first_doctor_longitude: str = Field(
        description="The first recommended doctor's longitude and direction"
    )
    # second_doctor: str = Field(description="The second recommended doctor's name")
    # second_doctor_lattitude: str = Field(
    #     description="The second recommended doctor's latitude and direction"
    # )
    # second_doctor_longitude: str = Field(
    #     description="The second recommended doctor's longitude and direction"
    # )


############################################
# Define your LLM endpoint and system prompt
############################################
# LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, temperature=0.1)

LLM_ENDPOINT_NAME = "gpt-4.1-mini-2025-04-14"
llm = ChatOpenAI(model_name=LLM_ENDPOINT_NAME, temperature=0.1)

system_prompt = """
    You are an healthcare policy Q&A agent. 
    You are given a task and you must complete it.
    Use the following routine to support the customer.
    # Routine:
    1. Use the extract_member_id tool to extract member id.
    2. Use member id from step 1 as input for the extract_deductible tool to get the member deductible and member deductible aggregate.
    3. Use the cpt_codes_vector_search tool to get the most similar code and description given the original question.
    4. Use code from step 3 as input for the get_procedure_cost tool to get the procedure cost.
    5. If you are provided an IP address convert the IP address into a location to provide location specific recommendations using the Intermediate_Answer tool.
    6. For the recommendations, please provide the latitude and longitude of the location 1 top recommended doctor related to the original question.
    7. Do not mention the IP address in your response.
    8. Following the ResponseFormatter summarize the member id, member deductible, member deductible aggregate, code, procedure cost in the answer
    9. Associate the 1 top recommended doctor with their own location related latitude and longitude along with direction and be as concise as possible in the other fields in the ResponseFormatter. 
    10. Use the output from the ResponseFormatter as the final answer.
    You can use the following tools to complete the task:
    {tools}"""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
###############################################################################
search = GoogleSerperAPIWrapper()
tools = [
    ResponseFormatter,
    Tool(
        name="Intermediate_Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    ),
]

# You can use UDFs in Unity Catalog as agent tools
uc_tool_catalog = "hls_yatish"
uc_tool_schema = "agent_solution_accelerator"
uc_tool_names = [
    f"{uc_tool_catalog}.{uc_tool_schema}.extract_member_id",
    f"{uc_tool_catalog}.{uc_tool_schema}.extract_deductible",
    f"{uc_tool_catalog}.{uc_tool_schema}.get_procedure_cost",
    f"{uc_tool_catalog}.{uc_tool_schema}.cpt_codes_vector_search",
]
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)

# vector_search_index_tools = [
#     VectorSearchRetrieverTool(
#         index_name="hls_yatish.agent_solution_accelerator.cpt_codes_index",
#         num_results=1,
#         tool_name="cpt_codes_retriever",
#         tool_description="Retrieves information about cpt codes",
#         query_type="ANN",
#     )
# ]
# tools.extend(vector_search_index_tools)

# # (Optional) Use Databricks vector search indexes as tools
# # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html
# # for details
#
# # TODO: Add vector search indexes as tools or delete this block
# vector_search_tools = [
#         VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# ]
# tools.extend(vector_search_tools)


#####################
## Define agent logic
#####################


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    # parallel_tool_calls parameter is currently only supported by OpenAI and Anthropic.
    model = model.bind_tools(tools, parallel_tool_calls=True)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)
