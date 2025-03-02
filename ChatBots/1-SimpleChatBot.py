
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables from a .env file
load_dotenv()

# Set the GROQ API key from environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Import and initialize the ChatGroq model with a specified model
from langchain_groq import ChatGroq
model = ChatGroq(model="Gemma2-9b-It")

# Invoke the model with a sequence of messages
print("Invoking model with initial messages...")
model.invoke([
    HumanMessage(content="Hi , My name is Neel and I am a Chief AI Engineer"),
    AIMessage(content="Hello Neel! It's nice to meet you. \n\nAs a Chief AI Engineer, what kind of projects are you working on these days?\n\n"),
    HumanMessage(content="Hey What's my name and what do I do?")
])

# Dictionary to store chat session histories
store = {}

# Function to retrieve or create a chat message history for a given session ID
def get_Session_History(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create a chat message history for a given session ID."""
    if session_id not in store:
        print(f"Creating new session history for session ID: {session_id}")
        store[session_id] = ChatMessageHistory(messages=[])
    else:
        print(f"Retrieving existing session history for session ID: {session_id}")
    return store[session_id]

# Create a runnable with message history using the model and session history function
with_message_history = RunnableWithMessageHistory(model, get_Session_History)

# Configuration for the first session
config = {"configurable": {"session_id": "1"}}

# Invoke the model with a new message and session configuration
print("Invoking model with a new message for session 1...")
response = with_message_history.invoke(
    [HumanMessage(content="Hi , My name is Neel and professionals love me for my teaching")],
    config=config
)
print("Response for session 1:", response.content)

# Configuration for a second session
config1 = {"configurable": {"session_id": "chat2"}}

# Invoke the model with a new message and session configuration
print("Invoking model with a new message for session 2...")
response = with_message_history.invoke(
    [HumanMessage(content="Whats my name")],
    config=config1
)
print("Response for session 2:", response.content)

# Update the session with a new message
print("Updating session 2 with a new message...")
response = with_message_history.invoke(
    [HumanMessage(content="Hey My name is Neel")],
    config=config1
)

# Print the response content
print("Updated response for session 2:", response.content)
