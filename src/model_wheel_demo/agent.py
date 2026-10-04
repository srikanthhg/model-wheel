"""Minimal Databricks-hosted LLM agent."""

from databricks_langchain import ChatDatabricks
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DEFAULT_MODEL = "databricks-meta-llama-3-3-70b-instruct"


def build_agent(model_name: str = DEFAULT_MODEL) -> AgentExecutor:
    llm = ChatDatabricks(endpoint=model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise helpful assistant. Answer clearly."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    return AgentExecutor(
        agent=create_tool_calling_agent(llm, [], prompt),
        tools=[],
        verbose=False,
    )
