import os
from dotenv import load_dotenv
from ebooklib import epub
from bs4 import BeautifulSoup
from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    set_global_service_context,
    Document,
)
from llama_index.tools import QueryEngineTool
from llama_index.agent import OpenAIAgent
from langchain.chat_models import ChatOpenAI

# Suppress warnings (optional)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load OpenAI API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is missing!")

# Verify EPUB file exists
epub_path = 'birth_of_tragedy.epub'
if not os.path.exists(epub_path):
    raise FileNotFoundError(f"EPUB file not found at path: {epub_path}")

# Function to load and parse EPUB into LlamaIndex Documents
def load_epub_to_documents(epub_path):
    book = epub.read_epub(epub_path)
    documents = []
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            content = item.get_content()
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            text = text.replace('\n', ' ').replace('\r', ' ').strip()
            if text:
                doc = Document(text)
                documents.append(doc)
    return documents

# Load EPUB file into Documents
documents = load_epub_to_documents(epub_path)

# Initialize LLM Predictor and Service Context
llm_predictor = LLMPredictor(llm=ChatOpenAI(openai_api_key=api_key, temperature=0))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
set_global_service_context(service_context)

# Create a Vector Store Index from the Documents
index = GPTVectorStoreIndex.from_documents(documents)

# Create a Query Engine from the Index
query_engine = index.as_query_engine()

# Define a Tool that Uses the Query Engine
query_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata={
        'name': 'NietzscheBook',
        'description': (
            'Use this tool when you need to answer questions about "The Birth of Tragedy" by Nietzsche.'
        ),
    },
)

# Create an Agent that Can Decide Whether to Use the Tool
agent = OpenAIAgent.from_tools([query_tool], llm=llm_predictor.llm)

# Define the User Query
query = "What is the difference between the Apollonian and Dionysian?"

# Run the Agent with the Query
response = agent.chat(query)

# Print the Agent's Response
print("Agent's Response:")
print(response)
