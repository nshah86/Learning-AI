import os
from dotenv import load_dotenv
load_dotenv()
import io
from PIL import Image as PILImage

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="mixtral-8x7b-32768")

class State(TypedDict):
    topic:str
    joke:str
    story:str
    poem:str
    combined_content:str

def generate_joke(state:State):
    """Generate a joke"""
    msg = llm.invoke(f"Generate a joke about {state['topic']}")
    return {"joke": msg.content}

def generate_story(state:State):
    """Generate a story"""
    msg = llm.invoke(f"Generate a story about {state['topic']}")
    return {"story": msg.content}   

def generate_poem(state:State):
    """Generate a poem"""
    msg = llm.invoke(f"Generate a poem about {state['topic']}")
    return {"poem": msg.content}

def aggregate_content(state:State):
    """Combine the joke, story, and poem into one string"""
    combined = (
        f"Here's a story, joke, and poem about {state['topic']}:\n\n"
        f"Story: {state['story']}\n\n"
        f"Joke: {state['joke']}\n\n"
        f"Poem: {state['poem']}\n\n"
    )
    return {"combined_content": combined}

# Build the workflow
parallel_workflow = StateGraph(State)

# Add nodes
parallel_workflow.add_node("generate_joke", generate_joke)
parallel_workflow.add_node("generate_story", generate_story)
parallel_workflow.add_node("generate_poem", generate_poem)
parallel_workflow.add_node("aggregate_content", aggregate_content)

# Add edges
parallel_workflow.add_edge(START, "generate_joke")
parallel_workflow.add_edge(START, "generate_story")
parallel_workflow.add_edge(START, "generate_poem")
parallel_workflow.add_edge("generate_joke", "aggregate_content")
parallel_workflow.add_edge("generate_story", "aggregate_content")
parallel_workflow.add_edge("generate_poem", "aggregate_content")
parallel_workflow.add_edge("aggregate_content", END)

# Compile the graph
graph = parallel_workflow.compile()

# Invoke the graph
result = graph.invoke({"topic": "Space Exploration"})
print(result)

# Attempt to display or save the workflow graph
try:
    from IPython.display import Image, display
    graph_bytes = parallel_workflow.get_graph().draw_mermaid_png()
    display(Image(graph_bytes))
except ImportError:
    # Fallback for non-IPython environments
    graph_bytes = parallel_workflow.get_graph().draw_mermaid_png()
    with open("parallel_workflow.png", "wb") as f:
        f.write(graph_bytes)
    print("Workflow graph saved as parallel_workflow.png")
