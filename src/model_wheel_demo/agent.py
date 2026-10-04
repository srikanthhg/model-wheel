# """Minimal Databricks-hosted LLM agent."""

# from databricks_langchain import ChatDatabricks
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# DEFAULT_MODEL = "databricks-meta-llama-3-3-70b-instruct"


# def build_agent(model_name: str = DEFAULT_MODEL) -> AgentExecutor:
#     llm = ChatDatabricks(endpoint=model_name)

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a concise helpful assistant. Answer clearly."),
#             ("human", "{input}"),
#             MessagesPlaceholder(variable_name="agent_scratchpad"),
#         ]
#     )

#     return AgentExecutor(
#         agent=create_tool_calling_agent(llm, [], prompt),
#         tools=[],
#         verbose=False,
#     )
"""Databricks-hosted LLM agent with a custom wheel tool."""

from databricks_langchain import ChatDatabricks
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from .greeting import greeting

DEFAULT_MODEL = "databricks-meta-llama-3-3-70b-instruct"


@tool
def greeting_tool(name: str) -> str:
    """Return a greeting using the model-wheel-demo Python library.

    Use this tool when the user asks to greet someone or asks for a
    personalized greeting.
    """
    return greeting(name)


def build_agent(model_name: str = DEFAULT_MODEL) -> AgentExecutor:
    llm = ChatDatabricks(endpoint=model_name)

    tools = [greeting_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise helpful assistant. "
                "Use the available tools when appropriate.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    return AgentExecutor(
        agent=create_tool_calling_agent(llm, tools, prompt),
        tools=tools,
        verbose=False,
    )
