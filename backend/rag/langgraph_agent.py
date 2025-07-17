import os
from langgraph.graph import StateGraph, END
# from langgraph.nodes import Node
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import Dict, Any,TypedDict,Optional,List
from pydantic import SecretStr

# --- Node Definitions ---

class AgentState(TypedDict, total=False):
    user_message: str
    retriever: Any
    llm: Any
    retrieved_docs: Optional[List[Document]]
    llm_response: Optional[str]
    intent: Optional[str]


def classify_node(state: AgentState) -> AgentState:
    """Classify user intent: product, meta-query, or invalid."""
    user_message = state.get("user_message", "")
    # Simple rule-based classifier for demo; replace with LLM if needed
    if any(x in user_message.lower() for x in ["fabric", "rug", "sofa", "curtain", "product", "furniture"]):
        state["intent"] = "product"
    elif any(x in user_message.lower() for x in ["help", "about", "who are you", "catalog", "what can you do"]):
        state["intent"] = "meta"
    else:
        state["intent"] = "invalid"
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from the vector store."""
    retriever = state.get("retriever")
    user_message = state.get("user_message", "")
    docs = retriever.retrieve(user_message, k=20) if retriever else []
    state["retrieved_docs"] = docs
    return state


def reason_node(state: AgentState) -> AgentState:
    """Generate a final answer using the LLM, grounded in retrieved docs."""
    llm = state.get("llm")
    docs = state.get("retrieved_docs") or []
    user_message = state.get("user_message", "")
    context = "\n".join([doc.page_content for doc in docs])
    prompt = (
    "You are a strict, helpful assistant for a home decor product catalog.\n"
    "Use ONLY the provided context to answer. If the context does NOT contain relevant information, do NOT attempt to answer.\n"
    "If the context lacks sufficient detail, ask a clear follow-up question instead of assuming.\n"
    "\n"
    "If enough details are found, reply STRICTLY as a JSON object structured like this:\n"
    "{{\n"
    "  'type': 'product_suggestion',\n"
    "  'summary': 'Summary of suggestions.',\n"
    "  'products': [\n"
    "    {{'name': 'Product Name', 'price': 'Price', 'url': 'Product URL'}},\n"
    "    ...\n"
    "  ]\n"
    "}}\n"
    "\n"
    "DO NOT reply in free text.\n"
    "\n"
    "Context:\n{context}\n\n"
    "User Message:\n{user_message}\n\n"
    "Assistant:"
)
    full_prompt = prompt.format(context=context, user_message=user_message)
    response = llm.predict(full_prompt) if llm else ""
    state["llm_response"] = response
    return state


def clarify_node(state: AgentState) -> AgentState:
    """Ask for missing info (e.g., room type, budget)."""
    llm = state.get("llm")
    user_message = state.get("user_message", "")
    prompt = (
        "You are a helpful assistant specialized in home decor products.\n"
        "If the user's request is missing important details (like room type, budget, preferred style, or quantity), ask a clear, concise follow-up question to gather that missing information.\n"
        "Do NOT attempt to answer unless required details are available.\n"
        "\n"
        "Example Follow-up:\n"
        "What type of room is this product for? Or do you have a preferred budget range?\n"
        "\n"
        "User Request:\n{user_message}\n\n"
        "Your follow-up question:"
    )
    response = llm.predict(prompt.format(user_message=user_message)) if llm else ""
    state["llm_response"] = response
    return state


def reject_node(state: AgentState) -> AgentState:
    """Respond with fallback for irrelevant queries."""
    state["llm_response"] = (
        "Sorry, I can only help with home decor product queries like fabrics, rugs, furniture based on our catalog."
    )
    return state

def meta_node(state: AgentState) -> AgentState:
    state['llm_response'] = (
        "I’m a virtual assistant trained to help with home decor products like rugs, sofas, and curtains. "
        "You can ask me about product options, prices, or designs."
    )
    return state



# --- LangGraph Workflow ---


def build_langgraph_agent(retriever, openai_api_key: str):
    llm = ChatOpenAI(api_key=SecretStr(openai_api_key), temperature=0)

    state_schema = AgentState
    graph = StateGraph(state_schema)

    # Register nodes
    # graph.add_node(Node("classify", classify_node))
    # graph.add_node(Node("retrieve", retrieve_node))
    # graph.add_node(Node("reason", reason_node))
    # graph.add_node(Node("clarify", clarify_node))
    # graph.add_node(Node("reject", reject_node))

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("reject", reject_node)
    graph.add_node("meta", meta_node)

    # Edges: classify -> retrieve/reject/clarify/meta
    # graph.add_edge("classify", "retrieve", condition=lambda s: s["intent"] == "product")
    # graph.add_edge("classify", "meta", condition=lambda s: s["intent"] == "meta")
    # graph.add_edge("classify", "reject", condition=lambda s: s["intent"] == "invalid")

    graph.add_conditional_edges(
    "classify",
    lambda s: s["intent"],
        {
            "product": "retrieve",
            "meta": "meta",
            "invalid": "reject",
        },
    )

    # retrieve -> reason
    graph.add_edge("retrieve", "reason")
    # reason -> END
    graph.add_edge("reason", END)
    # clarify -> END
    graph.add_edge("clarify", END)
    # reject -> END
    graph.add_edge("reject", END)

    # Set start node
    graph.set_entry_point("classify")
    app = graph.compile()

    def run_agent(user_message: str):
        state:AgentState = {
            "user_message": user_message,
            "retriever": retriever,
            "llm": llm,
        }
        # Classify intent
        state = classify_node(state)
        # Run the graph
        result = app.invoke(state)
        return result.get("llm_response", "Sorry, I couldn't process your request.")

    return run_agent
