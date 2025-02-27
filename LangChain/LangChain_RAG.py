import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set USER_AGENT environment variable
# This is used to identify the application making requests , you will set up in LANGCHAIN website when you create an account and get the API keys
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "Neel_Learn/1.0.0")

# Set other environment variables
# These are used for authentication and configuration
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Import remaining dependencies
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the language model
# Using gpt-4o-mini with a temperature of 0 for deterministic responses
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

try:
    # Load and split documents
    loader = WebBaseLoader("https://cricinfo.com/")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)
    print("Documents loaded and split successfully!")
    print(f"Number of documents: {len(documents)}")

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("Vector store created successfully!")

    # Optional: Test a simple query
    query = "What is the latest cricket news?"
    docs = vectorstore.similarity_search(query, k=2)
    print("\nTest Query Results:")
    for doc in docs:
        print("\nContent:", doc.page_content[:200], "...") 
        prompt = ChatPromptTemplate.from_template(
            "Answer the following question based on the context provided: {context}\n\nQuestion: {question}\nAnswer:"
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        print(document_chain.invoke({"context": docs, "question": query}))
except Exception as e:
    print(f"Error occurred: {str(e)}")



