from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from backend.rag.retriever import ExcelRetriever
from backend.rag.langgraph_agent import build_langgraph_agent
from typing import Optional 

# Load environment variables
load_dotenv("./backend/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

if OPENAI_API_KEY is None:
    raise ValueError("Missing OpenAI API Key. Please set OPENAI_API_KEY in your environment.")

app = FastAPI()

# Allow CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# Initialize retriever and LangGraph agent at startup
retriever = ExcelRetriever(DATA_DIR, OPENAI_API_KEY)
langgraph_agent = build_langgraph_agent(retriever, OPENAI_API_KEY)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    response = langgraph_agent(user_message)
    return {"response": response} 