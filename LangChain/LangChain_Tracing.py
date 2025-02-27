import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Langsmith Tracking and tracing

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")


llm=ChatOpenAI(model="gpt-3.5-turbo")
print(llm)
result=llm.invoke("What is agentic AI")
print(result.content)
