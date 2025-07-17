import os
import pandas as pd
from glob import glob
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data_source", "faiss_index")
os.makedirs(INDEX_DIR, exist_ok=True)

def get_all_excel_files(data_dir):
    return glob(os.path.join(data_dir, '**', '*.xlsx'), recursive=True)

def build_faiss_index():
    all_docs = []
    for file_path in get_all_excel_files(DATA_DIR):
        df = pd.read_excel(file_path)
        for i, row in df.iterrows():
            doc = Document(
                page_content=' | '.join(map(str, row)),
                metadata={"row": i, "file": file_path}
            )
            all_docs.append(doc)
    print(f"Loaded {len(all_docs)} documents from Excel files.")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split into {len(split_docs)} chunks for embedding.")
    
    if OPENAI_API_KEY is None:
        raise ValueError("Missing OpenAI API Key. Please set OPENAI_API_KEY in your environment.")

    embeddings = OpenAIEmbeddings(api_key=SecretStr(OPENAI_API_KEY))
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"FAISS index saved to {INDEX_DIR}")

if __name__ == "__main__":
    build_faiss_index() 