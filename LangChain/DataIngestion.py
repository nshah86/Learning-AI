#Documentation: https://python.langchain.com/docs/modules/data_connection/document_loaders/
#PDF loader: https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
#Text loader: https://python.langchain.com/docs/modules/data_connection/document_loaders/text

import os
from dotenv import load_dotenv
from pprint import pprint
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader 
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import WikipediaLoader

import bs4 #BeautifulSoup
#Load the environment variables
load_dotenv()

print("--------------Text Loader------------------")        
#Text Loader
loader=TextLoader('speech.txt') #path to the text file
text_documents=loader.load()
text_documents
pprint(text_documents)
print("---------------END Text Loader -----------------")

print("--------------PDF Loader------------------")
#PDF Loader
loader=PyPDFLoader('syllabus.pdf') #path to the pdf file
pdf_documents=loader.load()
pdf_documents
pprint(pdf_documents)
print("---------------END PDF Loader -----------------")

print("--------------Web Loader------------------")
#Web Loader
loader=WebBaseLoader('https://www.cricinfo.com') #url of the website    
web_documents=loader.load()
web_documents
pprint(web_documents)   
print("---------------END Web Loader -----------------")

print("--------------Arxiv Loader------------------")
#Arxiv
loader = ArxivLoader(query="1706.03762", load_max_docs=2) #query is the arxiv id of the paper
arxiv_documents=loader.load()
arxiv_documents
pprint(arxiv_documents)
print("---------------END Arxiv Loader -----------------")

print("--------------Wikipedia Loader------------------")
#Wikipedia
loader = WikipediaLoader(query="LangChain", load_max_docs=2) #query is the title of the wikipedia page
wikipedia_documents=loader.load()
wikipedia_documents
pprint(wikipedia_documents)
print("---------------END Wikipedia Loader -----------------")

