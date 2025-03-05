#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangGraph Agents Implementation

This script demonstrates the implementation of a ReAct-style agent using LangGraph.
ReAct is a general agent architecture that follows these steps:
- Act: Let the model call specific tools
- Observe: Pass the tool output back to the model
- Reason: Let the model reason about the tool output to decide what to do next

The script includes:
1. Basic setup with mathematical tools (add, multiply, divide)
2. LangGraph implementation of the agent workflow
3. Memory management for maintaining conversation context
4. Visualization of the agent graph structure
"""

import os
import tempfile
from dotenv import load_dotenv
from IPython.display import Image, display

# Load environment variables for API keys
load_dotenv()

# Set API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Import required libraries
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from typing import Annotated, Dict, List, Any
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
# Instead of using MemorySaver, we'll use a simple dictionary to maintain conversation state
# from langgraph.saver import MemorySaver

# Define mathematical tools that will be used by the agent
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

# Create a list of tools to be used by the agent
tools = [add, multiply, divide]

# Initialize the language model with tools
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# Define the state structure for our graph
# This uses TypedDict to enforce the structure of our state
class MessagesState(TypedDict):
    """State consisting of messages."""
    messages: Annotated[list[AnyMessage], add_messages]

# Create the assistant function that processes messages and generates responses
def assistant(state: MessagesState) -> Dict[str, List[AnyMessage]]:
    """
    This function processes the current state (messages) and generates a response
    using the language model with tools.
    
    Args:
        state: The current state containing messages
        
    Returns:
        A dictionary with updated messages
    """
    # Create a system message to instruct the assistant
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
    
    # Invoke the LLM with the system message and current conversation history
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Now we'll build our graph
def build_graph():
    """
    Build the LangGraph for our agent.
    
    Returns:
        A compiled graph that can process messages
    """
    # Initialize the state graph with our MessagesState
    builder = StateGraph(MessagesState)
    
    # Define the nodes in our graph
    builder.add_node("assistant", assistant)  # The LLM that can use tools
    builder.add_node("tools", ToolNode(tools))  # Node for executing tools
    
    # Define the edges that connect our nodes
    
    # Start -> Assistant: Start by sending the user's message to the assistant
    builder.add_edge(START, "assistant")
    
    # Assistant -> Next node (conditional):
    # If assistant calls a tool -> go to tools node
    # If assistant replies directly -> exit the graph
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    
    # Tools -> Assistant: After tools execute, send results back to assistant
    builder.add_edge("tools", "assistant")
    
    # Compile the graph
    react_graph = builder.compile()
    
    return react_graph

# Build our agent graph without memory (default implementation)
react_graph = build_graph()

# Visualize the graph structure
def print_graph():
    """
    Generate and display a visualization of the agent graph.
    Saves the graph image to 'langgraph_agent_diagram.png' in the current directory.
    """
    # Generate the graph image
    graph_image = react_graph.get_graph().draw_mermaid_png()
    
    # Save the image to a file in the current directory
    output_path = "langgraph_agent_diagram.png"
    try:
        with open(output_path, "wb") as f:
            f.write(graph_image)
        print(f"Graph visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error saving graph image: {e}")
        
        # Fallback to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            with open(tmp_path, "wb") as f:
                f.write(graph_image)
            print(f"Graph visualization saved to temporary file: {tmp_path}")
    
    # In environments that support display (like notebooks), show the image
    try:
        display(Image(graph_image))
        print("Graph visualization displayed.")
    except Exception as e:
        print(f"Note: Graph image was saved but cannot be displayed in current environment: {e}")

# Helper function to print messages in a consistent way
def print_messages(messages, title=None):
    """
    Print a list of messages in a standardized format.
    
    Args:
        messages: List of messages to print
        title: Optional title to display before messages
    """
    if title:
        print(f"\n{title}")
        print("=" * 50)
    
    for m in messages:
        print(f"[{m.type}]: {m.content}")
        
        # Handle tool calls safely
        if hasattr(m, 'tool_calls') and m.tool_calls:
            try:
                for tool_call in m.tool_calls:
                    # Try to extract tool information regardless of format
                    if hasattr(tool_call, 'name') and hasattr(tool_call, 'args'):
                        print(f"  Tool Call: {tool_call.name}{tool_call.args}")
                    elif isinstance(tool_call, dict):
                        if 'name' in tool_call:
                            args = tool_call.get('args', {})
                            print(f"  Tool Call: {tool_call['name']}{args}")
                        else:
                            print(f"  Tool Call: {tool_call}")
                    else:
                        print(f"  Tool Call: {tool_call}")
            except Exception as e:
                print(f"  Error displaying tool calls: {str(e)}")
                
        # Display tool results
        if m.type == 'tool' and hasattr(m, 'name'):
            print(f"  Tool Result: {m.content}")
        
        print("-" * 50)

# ========================
# Testing the Basic Agent
# ========================

def run_agent_example():
    """
    Run an example query through the agent that demonstrates multi-step reasoning.
    """
    print("Running example query through the agent...")
    # Create a message asking for a sequence of mathematical operations
    messages = [HumanMessage(content="Add 10 and 14. Multiply the output by 2. Divide the output by 5")]
    
    # Process the message through our agent graph
    result = react_graph.invoke({"messages": messages})
    
    # Print each message in the conversation using our helper function
    print_messages(result['messages'], "Full conversation")

# ========================
# Memory in Agents
# ========================

def demonstrate_memory_limitation():
    """
    Demonstrate the memory limitation of the basic implementation.
    """
    print("\n==== Demonstrating Memory Limitation ====")
    
    # First message
    messages1 = [HumanMessage(content="Add 14 and 15.")]
    result1 = react_graph.invoke({"messages": messages1})
    
    # Print the first interaction
    print_messages(result1['messages'], "First interaction")
    
    # Second message asking to use previous result - will fail without memory
    messages2 = [HumanMessage(content="Multiply that by 2.")]
    result2 = react_graph.invoke({"messages": messages2})
    
    # Print the second interaction
    print_messages(result2['messages'], "Second interaction (without memory)")

# Simple in-memory conversation store
conversation_memory = {}

def demonstrate_memory_implementation():
    """
    Demonstrate how to implement and use memory in the agent using a manual approach.
    """
    print("\n==== Demonstrating Memory Implementation  ====")
    
    # Create a unique thread ID for this conversation
    thread_id = "thread_1"
    
    # Initialize the thread's memory if it doesn't exist
    if thread_id not in conversation_memory:
        conversation_memory[thread_id] = []
    
    # First message with memory
    initial_message = HumanMessage(content="Add 13 and 14.")
    
    # Add this message to our conversation memory
    conversation_memory[thread_id].append(initial_message)
    
    # Process the message through our agent graph
    result1 = react_graph.invoke({"messages": conversation_memory[thread_id]})
    
    # Update our conversation memory with the new messages (excluding the initial ones)
    new_messages1 = result1["messages"][len(conversation_memory[thread_id]):]
    conversation_memory[thread_id].extend(new_messages1)
    
    # Print the first interaction
    print_messages(result1['messages'], "First interaction (with memory)")
    
    # Second message using memory to refer to previous result
    follow_up_message = HumanMessage(content="Multiply that by 2.")
    
    # Add this message to our conversation memory
    conversation_memory[thread_id].append(follow_up_message)
    
    # Process with the full conversation history
    result2 = react_graph.invoke({"messages": conversation_memory[thread_id]})
    
    # Update our conversation memory with just the new messages
    new_messages2 = result2["messages"][len(conversation_memory[thread_id]):]
    conversation_memory[thread_id].extend(new_messages2)
    
    # Print the second interaction
    print_messages(new_messages2, "Second interaction (with memory)")

if __name__ == "__main__":
    # Build and print the graph
    print("Building and visualizing the agent graph...")
    print_graph()
    
    # Run the example
    run_agent_example()
    
    # Demonstrate memory limitation
    demonstrate_memory_limitation()
    
    # Demonstrate memory implementation
    demonstrate_memory_implementation() 
