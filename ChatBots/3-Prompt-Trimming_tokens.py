import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage, trim_messages
from operator import itemgetter

# Load environment variables from a .env file
load_dotenv()  # Loading all the environment variables

# Set the GROQ API key from environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq model with a specified model
model = ChatGroq(model="Gemma2-9b-It")

# Dictionary to store chat session histories
store = {}

# Initialize a message trimmer to manage message length
trimmer = trim_messages(
    max_tokens=60,  # Maximum number of tokens allowed
    strategy="last",  # Strategy to trim messages, keeping the last messages
    token_counter=model,  # Model used to count tokens
    include_system=True,  # Include system messages in trimming
    allow_partial=False,  # Do not allow partial messages
    start_on="human"  # Start trimming on human messages
)

# Define a chat prompt template with a system message and a placeholder for messages
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define a sequence of messages for the conversation
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# Trim the messages to fit within the token limit
trimmer.invoke(messages)

# Create a chain to process messages through the prompt and model
chain = (RunnablePassthrough.assign(messages=itemgetter("messages")) | prompt | model)

# Invoke the chain with additional messages and language specification
response = chain.invoke({"messages": messages + [HumanMessage(content="whats icecream i like?")], "language": "English"})

# Print the response content
print(response.content)
