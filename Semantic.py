"""
pip3 install langchain langchain_community langchain_openai pymongo langchain_experimental python-dotenv
"""

from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")
llm_api_key = os.getenv("LLM_API_KEY")
db_name = "chunking_demo"
collection_name = "semantic_example"
embeddings = OpenAIEmbeddings(openai_api_key=llm_api_key)

create_collection = MongoClient(mongodb_uri)[db_name][collection_name]

with open("sample_files/text_example.txt") as f:
    text_example = f.read()

# Split the text into chunks based on semantic similarity    
text_splitter = SemanticChunker(OpenAIEmbeddings(openai_api_key=llm_api_key))
docs = text_splitter.create_documents([text_example])

# Initialize the vector store and insert the vectorized documents into the MongoDB Atlas Cluster
vector_store = MongoDBAtlasVectorSearch.from_documents(docs, embeddings, collection=create_collection)