"""
pip3 install langchain langchain_community langchain_openai pymongo python-dotenv
"""

from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.text_splitter import ( Language, RecursiveCharacterTextSplitter )
from dotenv import load_dotenv
import os

load_dotenv()

# Sample Python code to split into chunks
python_example_code = """
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# Example usage
person = Person("John", 25)
person.greet()
"""

mongodb_uri = os.getenv("MONGODB_URI")
llm_api_key = os.getenv("LLM_API_KEY")
db_name = "chunking_demo"
collection_name = "python_example"

create_collection = MongoClient(mongodb_uri)[db_name][collection_name]

embeddings = OpenAIEmbeddings(openai_api_key=llm_api_key)

# Split the Python code into chunks of 100 characters with 0 overlap
python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=100, chunk_overlap=0
)
python_docs = python_splitter.create_documents([python_example_code])

# Initialize the vector store and insert the vectorized documents into the MongoDB Atlas Cluster
vector_store = MongoDBAtlasVectorSearch.from_documents(python_docs, embeddings, collection=create_collection)