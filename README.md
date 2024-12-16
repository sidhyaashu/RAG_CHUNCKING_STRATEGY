### RECURSIVE
```plaintext
Prerequisites
A sample PDF document (example available in the Curriculum GitHub repository)
A .env to file containing an Atlas Cluster Connection String and OpenAI API Key
Install requirements:
pip3 install langchain langchain_community langchain_openai pymongo pypdf python-dotenv 


Usage
recursive_splitter.py file
The following code initializes a MongoDB collection and loads a PDF file, then splits the PDF content into chunks of 50 characters with no overlap using a recursive character text splitter. It then vectorizes these chunks using OpenAI embeddings. Finally, it stores the vectorized documents in a MongoDB Atlas cluster.

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

Run the demo:
python3 recursive_splitter.py

```






### DOCUMENT SPECIFIC
```plaintext
Prerequisites
A .env to file containing an Atlas Cluster Connection String and OpenAI API Key
Install requirements:
pip3 install langchain langchain_community langchain_openai pymongo python-dotenv


Usage
python_splitter.py file
The following code connects to a MongoDB Atlas cluster and creates a collection named “python_example” within the “chunking_demo” database. It uses the RecursiveCharacterTextSplitter to split a given Python code example into chunks of 100 characters with no overlap. These chunks are then vectorized using OpenAI embeddings and stored in the MongoDB collection for vector search purposes.

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

Run the demo:
python3 python_splitter.py
```





### SEMANTIC 
```plaintext
Prerequisites
A sample txt document (example available in the Curriculum GitHub repository)
A .env to file containing an Atlas Cluster Connection String and OpenAI API Key
Install requirements:
pip3 install langchain langchain_community langchain_openai pymongo langchain_experimental python-dotenv


Usage
semantic_splitter.py file
The following code initializes a MongoDB collection and reads from a txt file. Then the SemanticChunker method is used to split the text into semantically similar chunks. Finally, it initializes a vector store and inserts the vectorized documents into a MongoDB Atlas cluster.

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

Run the demo:
python3 semantic_splitter.py

```