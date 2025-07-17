from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
import os

class ReActRAGAgent:
    def __init__(self, retriever, openai_api_key: str):
        self.retriever = retriever
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
        self.agent = self._build_agent()

    def _build_agent(self):
        # Define a tool for retrieval
        def retrieve_tool(query: str):
            docs = self.retriever.retrieve(query)
            return '\n'.join([doc.page_content for doc in docs])
        tools = [
            Tool(
                name="ProductRetriever",
                func=retrieve_tool,
                description="Useful for answering questions about decor product fabrics. Always use this to find information."
            )
        ]
        # Initialize a ReAct agent
        agent = initialize_agent(
            tools,
            self.llm,
            agent="chat-conversational-react-description",
            verbose=True
        )
        return agent

    def run(self, query: str, chat_history=None):
        # Optionally pass chat_history for context
        return self.agent.run(input=query) 