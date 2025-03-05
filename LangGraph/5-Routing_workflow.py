# message_routing.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from PIL import Image as PILImage
import io

# Load environment variables from a .env file
load_dotenv()

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define a custom type for message state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define functions to be used as tools in the message routing system
def route_message(destination: str, message: str) -> str:
    """Route a message to a specific destination.

    Args:
        destination: The destination to route the message to
        message: The message to be routed
    """
    return f"Message '{message}' routed to {destination}."

def log_message(message: str) -> str:
    """Log a message for auditing purposes.

    Args:
        message: The message to be logged
    """
    return f"Message '{message}' logged for auditing."

# List of tools (functions) to be used
tools = [route_message, log_message]

# Initialize the language model with OpenAI's GPT-4o
llm = ChatOpenAI(model="gpt-4o")

# Bind the tools to the language model
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# Define a system message to guide the assistant's behavior
sys_msg = SystemMessage(content="You are a system responsible for routing and logging messages.")

# Define the assistant function to process messages
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Initialize a state graph for managing message flow
builder = StateGraph(MessagesState)

# Define nodes in the graph
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges in the graph
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)

builder.add_edge("tools", "assistant")

# Compile the graph
react_graph = builder.compile()

# Define initial messages and invoke the graph
messages = [HumanMessage(content="Route the message 'Hello World' to 'Support Team'.")]
messages = react_graph.invoke({"messages": messages})

# Print the messages
for m in messages['messages']:
    m.pretty_print()

# Generate and display the graph image using Pillow
# Convert the graph to an image
graph_image = react_graph.get_graph().draw_mermaid_png()

# Use Pillow to open and show the image
image = PILImage.open(io.BytesIO(graph_image))
image.show()

