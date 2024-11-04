import logging
import sys
import os

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
  SummaryExtractor,
  TitleExtractor,
  KeywordExtractor,
)
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
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

reader = SimpleDirectoryReader(input_dir=f"../documents/{collection_name}/", recursive=True)
documents = reader.load_data(show_progress=True)

db = chromadb.PersistentClient(path="./db")
chroma_collection = db.get_or_create_collection(collection_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

transformations = [
  SentenceSplitter(chunk_size=200, chunk_overlap=0),
  TitleExtractor(show_progress=True, llm=llm),
  SummaryExtractor(summaries=["prev", "self"], show_progress=True, llm=llm),
  KeywordExtractor(show_progress=True, llm=llm),
  embed_model,
]

pipeline=IngestionPipeline(
  transformations=transformations,
  vector_store=vector_store,
)

pipeline.run(documents=documents)

print("Documents embedded and stored in vector store.")

# from llama_index.core import StorageContext, VectorStoreIndex

# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, embed_model=embed_model
# )