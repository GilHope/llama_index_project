import os
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from dotenv import load_dotenv
import logging
import sys

# Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)


# Load OpenAI API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is missing!")

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents_beyond = SimpleDirectoryReader("books/beyond/").load_data()
    index_beyond = VectorStoreIndex.from_documents(documents_beyond)
    # store it for later
    index_beyond.storage_context.persist(persist_dir=PERSIST_DIR + "/beyond")

    documents_twilight = SimpleDirectoryReader("books/twilight/").load_data()
    index_twilight = VectorStoreIndex.from_documents(documents_twilight)
    # store it for later
    index_twilight.storage_context.persist(persist_dir=PERSIST_DIR + "/twilight")

    documents_tragedy = SimpleDirectoryReader("books/tragedy/").load_data()
    index_tragedy = VectorStoreIndex.from_documents(documents_tragedy)
    # store it for later
    index_tragedy.storage_context.persist(persist_dir=PERSIST_DIR + "/tragedy")

    documents_anti = SimpleDirectoryReader("books/anti/").load_data()
    index_anti = VectorStoreIndex.from_documents(documents_anti)
    # store it for later
    index_anti.storage_context.persist(persist_dir=PERSIST_DIR + "/anti")
else:
    # load the existing index
    storage_context_beyond = StorageContext.from_defaults(persist_dir=PERSIST_DIR + "/beyond")
    index_beyond = load_index_from_storage(storage_context_beyond)
    storage_context_twilight = StorageContext.from_defaults(persist_dir=PERSIST_DIR + "/twilight")
    index_twilight = load_index_from_storage(storage_context_twilight)
    storage_context_tragedy = StorageContext.from_defaults(persist_dir=PERSIST_DIR + "/tragedy")
    index_tragedy = load_index_from_storage(storage_context_tragedy)
    storage_context_anti = StorageContext.from_defaults(persist_dir=PERSIST_DIR + "/anti")
    index_anti = load_index_from_storage(storage_context_anti)

# documents = SimpleDirectoryReader("books/").load_data()
# index = VectorStoreIndex.from_documents(documents)
# query_engine = index_tragedy.as_query_engine()
# response = query_engine.query("Apollonian and Dionysian")
# print(response)

from llama_index.core.tools import QueryEngineTool, ToolMetadata

individual_query_engine_tools = [
    QueryEngineTool(
        query_engine = index_tragedy.as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_tragedy",
            description=f"useful for when you want to answer qustions about the Birth of Tragedy by Friedrich Nietzsche",
        ),
    ),
    QueryEngineTool(
        query_engine = index_twilight.as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_twilight",
            description=f"useful for when you want to answer qustions about the Twilight of the Idols by Friedrich Nietzsche",
        ),
    ),
    QueryEngineTool(
        query_engine = index_beyond.as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_beyond",
            description=f"useful for when you want to answer qustions about the Beyond Good and Evil by Friedrich Nietzsche",
        ),
    ),
    QueryEngineTool(
        query_engine = index_anti.as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_beyond",
            description=f"useful for when you want to answer qustions about the Antichrist by Friedrich Nietzsche",
        ),
    ),
]

from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    llm=OpenAI(),
)

# Sub query question engine
query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="useful for when you want to answer queries that require analyzing multiple books written by Nietzsche",
    ),
)

tools = individual_query_engine_tools + [query_engine_tool]

from llama_index.agent.openai import OpenAIAgent

agent = OpenAIAgent.from_tools(tools, verbose=True)

# while True:
#     text_input = input("User: ")
#     if text_input == "exit":
#         break
#     response = agent.chat(text_input)
#     print(f"Agent: {response}")

# response = query_engine.query("How does Nietzsche's concept of duality evolve in his works?")
# print(response)

# response = agent.chat("hi, i am bob")
# print(str(response))

# response = agent.chat("What is my name?")
# print(str(response))

response = agent.chat("What is the Apollonian and Dionysian?")
print(str(response))