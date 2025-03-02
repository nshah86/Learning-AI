import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter, RecursiveJsonSplitter
from langchain.document_loaders import PyPDFLoader
import json
import requests

# Initialize pprint for better output formatting
pp = pprint.PrettyPrinter(indent=4)

# Recursive Character Text Splitter
# This text splitter is the recommended one for generic text. It is parameterized by a list of characters.
# It tries to split on them in order until the chunks are small enough.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Example usage with PDF documents 
# only first 4 pages    

loader = PyPDFLoader('ANYPDF.pdf')
docs = loader.load()
final_documents = text_splitter.split_documents(docs[:4])


pp.pprint("this document split")
pp.pprint(final_documents)

# HTML Header Text Splitter
# A structure-aware chunker that splits text at the HTML element level and adds metadata for each header.
# Example HTML string
html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Foo</h1>
        <p>Some intro text about Foo.</p>
        <div>
            <h2>Bar main section</h2>
            <p>Some intro text about Bar.</p>
            <h3>Bar subsection 1</h3>
            <p>Some text about the first subtopic of Bar.</p>
            <h3>Bar subsection 2</h3>
            <p>Some text about the second subtopic of Bar.</p>
        </div>
        <div>
            <h2>Baz</h2>
            <p>Some text about Baz</p>
        </div>
        <br>
        <p>Some concluding text about Foo</p>
    </div>
</body>
</html>
"""

# Initialize HTMLHeaderTextSplitter with headers
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
pp.pprint("this HTML split")
pp.pprint(html_header_splits)

# Recursive JSON Splitter
# This JSON splitter splits JSON data while allowing control over chunk sizes.
json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()
json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = json_splitter.split_text(json_data)
pp.pprint("this JSON split")
pp.pprint(json_chunks)
