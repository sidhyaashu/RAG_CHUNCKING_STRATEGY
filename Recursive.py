"""
pip3 install langchain langchain_community langchain_openai pymongo pypdf python-dotenv 
"""

from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")
llm_api_key = os.getenv("LLM_API_KEY")
db_name = "chunking_demo"
collection_name = "recursive_example"
embeddings = OpenAIEmbeddings(openai_api_key=llm_api_key)

create_collection = MongoClient(mongodb_uri)[db_name][collection_name]

# Change the file path below if you are using a different PDF file with this demo
loader = PyPDFLoader('./sample_files/example.pdf')
data = loader.load()

# Split the PDF into chunks of 50 characters with 0 overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# Initialize the vector store and insert the vectorized documents into the MongoDB Atlas Cluster
vector_store = MongoDBAtlasVectorSearch.from_documents(docs, embeddings, collection=create_collection)