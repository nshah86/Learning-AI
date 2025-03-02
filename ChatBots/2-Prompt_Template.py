import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Load environment variables from a .env file
load_dotenv()  # Loading all the environment variables

# Set the GROQ API key from environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq model with a specified model
model = ChatGroq(model="Gemma2-9b-It")

# Dictionary to store chat session histories
store = {}

# Function to retrieve or create a chat message history for a given session ID
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Define a chat prompt template with a system message and a placeholder for messages
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all the questions to the best of your ability in {language}."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# Create a chain by combining the prompt and the model
Chain = prompt | model

# Create a runnable with message history using the chain and session history function
with_message_history = RunnableWithMessageHistory(Chain, get_session_history , input_messages_key="messages", output_messages_key="response")

# Configuration for a session with a specific session ID
config = {"configurable": {"session_id": "chat4"}}

# Invoke the model with a message and session configuration
print("Invoking model with a message in English...")
rresponse = with_message_history.invoke(
    {"messages": [HumanMessage(content="Hi, my name is Neel and I am a Chief AI Engineer")], "language": "English"},
    config=config
)
print(rresponse.content)

# Invoke the model with a message in a different language
print("Invoking model with a message in Hindi...")
rresponse = Chain.invoke({"messages": [HumanMessage(content="Hi, my name is Neel")], "language": "Hindi"})
print(rresponse.content)
