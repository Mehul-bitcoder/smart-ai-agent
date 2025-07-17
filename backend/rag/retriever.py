import pandas as pd
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os
from glob import glob
from pydantic import SecretStr

class ExcelRetriever:
    def __init__(self, data_dir: str, openai_api_key: str):
        self.data_dir = data_dir
        self.openai_api_key = openai_api_key
        self.index_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_source', 'faiss_index')
        self.index_path = os.path.join(self.index_dir, 'index.faiss')
        self.store_path = os.path.join(self.index_dir, 'index.pkl')
        print("Looking for FAISS index at:", self.index_path)
        self.retriever = self._load_retriever()

    def _load_retriever(self):
        embeddings = OpenAIEmbeddings(api_key=SecretStr(self.openai_api_key))
        if os.path.exists(self.index_path) and os.path.exists(self.store_path):
            print('Loading FAISS index from disk...')
            vectorstore = FAISS.load_local(self.index_dir, embeddings, allow_dangerous_deserialization=True)
            return vectorstore.as_retriever()
        else:
            raise FileNotFoundError(
                f"FAISS index not found in {self.index_dir}. Please run 'python backend/build_index.py' to build the index before starting the server."
            )

    def retrieve(self, query: str, k: int = 3):
        return self.retriever.get_relevant_documents(query, k=k) 