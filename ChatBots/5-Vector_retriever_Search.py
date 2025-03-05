# Document Initialization: Creates a list of Document objects with content and metadata.
# Language Model: Initializes a language model using ChatGroq.
# Embeddings: Uses HuggingFaceEmbeddings to create embeddings for the documents.
# Vector Store: Creates a vector store using FAISS and performs similarity searches.
# Retriever: Converts the vector store into a retriever and performs batch retrievals.
# RAG Chain: Sets up a retrieval-augmented generation chain using a prompt template and the retriever.i


import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize documents with content and metadata
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# Initialize the language model using ChatGroq
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

# Initialize embeddings using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a vector store from the documents using FAISS and embeddings
vectorstore = FAISS.from_documents(documents, embedding=embeddings)

# Perform a similarity search in the vector store
similar_docs = vectorstore.similarity_search("cat")
print("Documents similar to 'cat':", similar_docs)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Perform a batch retrieval
batch_results = retriever.batch(["cat", "dog"])
print("Batch retrieval results:", batch_results)

# Define a prompt template for retrieval-augmented generation (RAG)
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("human", message)])

# Create a RAG chain using the retriever and prompt
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

# Invoke the RAG chain with a question
response = rag_chain.invoke("tell me about dogs")
print("RAG response:", response.content)
