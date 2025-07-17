from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from rag.retriever import ExcelRetriever
from rag.react_agent import ReActRAGAgent

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

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
    session_id: str = None

# Initialize retriever and agent at startup
retriever = ExcelRetriever(DATA_DIR, OPENAI_API_KEY)
agent = ReActRAGAgent(retriever, OPENAI_API_KEY)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    # Optionally, use session_id for chat history
    response = agent.run(user_message)
    return {"response": response} 