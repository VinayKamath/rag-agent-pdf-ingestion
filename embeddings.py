from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Embeddings():
    def __init__(self, chunks):
        self.chunks = chunks
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
    
    def get_vectorstore(self):
        return self.vectorstore 