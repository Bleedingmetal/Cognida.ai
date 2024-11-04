# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
import os
import chromadb

collection_name = "visa"

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

llm = AzureOpenAI(
  engine="gpt-35-turbo-16k",
  api_key=api_key,
  azure_endpoint=azure_endpoint,
  api_version="2023-05-15",
)

embed_model = AzureOpenAIEmbedding(
  api_key=api_key,
  azure_endpoint=azure_endpoint,
  api_version="2023-05-15",
)

Settings.llm = llm
Settings.embed_model = embed_model

db = chromadb.PersistentClient(path="./db")
chroma_collection = db.get_or_create_collection(collection_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(
  vector_store=vector_store,
  embed_model=embed_model,
)

chat_engine = index.as_chat_engine()

while True:
  question = input("Enter question: ")
  response_stream = chat_engine.stream_chat(question)
  print("\nResponse: ")
  response_stream.print_response_stream()
  print("\n\n")