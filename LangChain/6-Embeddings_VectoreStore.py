import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Set API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Example 1: OpenAI Embeddings
openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
openai_embedding_result = openai_embeddings.embed_query("Sample text for OpenAI embeddings")
print("OpenAI Embeddings:", openai_embedding_result)

# Example 2: Ollama Embeddings
ollama_embeddings = OllamaEmbeddings(model="llama")
ollama_embedding_result = ollama_embeddings.embed_documents([
    "Alpha is the first letter of Greek alphabet",
    "Beta is the second letter of Greek alphabet"
])
print("Ollama Embeddings:", ollama_embedding_result)

# Example 3: Hugging Face Embeddings
huggingface_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
huggingface_embedding_result = huggingface_embeddings.embed_query("Sample text for Hugging Face embeddings")
print("Hugging Face Embeddings:", huggingface_embedding_result)

# Example 4: FAISS Vector Store
loader = TextLoader("speech.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
docs = text_splitter.split_documents(documents)

faiss_embeddings = OllamaEmbeddings()
faiss_db = FAISS.from_documents(docs, faiss_embeddings)

faiss_query = "How does the speaker describe the desired outcome of the war?"
faiss_docs = faiss_db.similarity_search(faiss_query)
print("FAISS Query Result:", faiss_docs[0].page_content)

# Example 5: Chroma Vector Store
chroma_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chroma_splits = chroma_text_splitter.split_documents(documents)

chroma_embeddings = OllamaEmbeddings()
chroma_db = Chroma.from_documents(documents=chroma_splits, embedding=chroma_embeddings)

chroma_query = "What does the speaker believe is the main reason the United States should enter the war?"
chroma_docs = chroma_db.similarity_search(chroma_query)
print("Chroma Query Result:", chroma_docs[0].page_content)
