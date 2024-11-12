from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents and build index
documents = SimpleDirectoryReader(
    "books/"
).load_data()
index = VectorStoreIndex.from_documents(documents)