import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from IPython.display import Image, display

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="mixtral-8x7b-32768")


class State(TypedDict):
    topic:str
    joke:str
    improved_joke:str
    final_joke:str

def generate_joke(state: State):
    """First LLM call to generate a joke"""
    msg = llm.invoke(f"Generate a joke about {state['topic']}")
    return {"joke": msg.content}

def check_punchline(state:State):
    """Gate function to check if the joke is funny"""
 # Simple check - does the joke contain "?" or "!"
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Fail"
    return "Pass"

def improve_joke(state:State):
    """Second LLM call to improve the joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}

def final_joke(state:State):
    """Final LLM call to generate the final joke"""
    msg = llm.invoke(f"Generate a final joke: {state['improved_joke']}")
    return {"final_joke": msg.content}


#Build the workflow
workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("generate_final_joke", final_joke)

#Add edges
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke",
    check_punchline,
    {"Fail":"improve_joke", "Pass":END}
)
workflow.add_edge("improve_joke", "generate_final_joke")
workflow.add_edge("generate_final_joke", END)

#Compile the graph
graph = workflow.compile()

#Invoke the graph
result = graph.invoke({"topic":"AI"})
print(result)
# Show workflow
display(Image(graph.get_graph().draw_mermaid_png()))

# Invoke the graph with a different topic
result = graph.invoke({"topic":"Space Exploration"})
print(result)
display(Image(graph.get_graph().draw_mermaid_png()))



