from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from IPython.display import Image, display
from PIL import Image as PILImage
import io

# Import necessary modules and classes
# - List and TypedDict for type annotations
# - StateGraph, START, END for graph building
# - HumanMessage for message handling
# - ChatPromptTemplate, StrOutputParser for prompt and output parsing
# - ChatGroq for the language model
# - os and dotenv for environment variable management
# - Image and display for displaying images

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Retrieve the GROQ API key from environment variables
# Raise an error if the API key is not set

llm = ChatGroq(
    model_name="gemma2-9b-it",
    api_key=groq_api_key,
    temperature=0.7
)

# Initialize the language model with specified parameters

class State(TypedDict):
    messages: List[HumanMessage]

# Define a State class for type annotations

graph_builder = StateGraph(State)

# Initialize a StateGraph with the State type

def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Define the chatbot function that takes a state and returns a new state with the response from the language model

graph_builder.add_node("chatbot", chatbot)

# Add a node to the graph for the chatbot function

graph_builder.add_edge(START, "chatbot")    
graph_builder.add_edge("chatbot", END)

# Add edges to the graph to define the flow from START to chatbot and from chatbot to END

graph = graph_builder.compile()

# Compile the graph

# Save the graph image to a file named 'chatbot_graph.png'
graph_image_data = graph.get_graph().draw_mermaid_png()
image = PILImage.open(io.BytesIO(graph_image_data))
image.save('chatbot_graph.png')

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Define a function to stream graph updates based on user input

while True:
    try: 
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting...")
            break
        stream_graph_updates(user_input)    
    except KeyboardInterrupt:
        user_input = "what do you think about langchain"
        print("User:", user_input)
        stream_graph_updates(user_input)
        break

# Continuously prompt the user for input and stream updates until the user exits or interrupts





