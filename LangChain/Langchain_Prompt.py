
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pprint import pprint
# Load environment variables
load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4")

# Create JSON output parser
output_parser = JsonOutputParser()

# Create a chat prompt template with JSON formatting instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI Engineer. Provide response in JSON format.\n{format_instructions}"),
    ("user", "{input}")
])

# Create the chain with format instructions
chain = prompt | llm | output_parser

# Example usage
response = chain.invoke({
    "input": "Can you tell me about Best options trading strategy?",
    "format_instructions": output_parser.get_format_instructions()
})
pprint(response)
