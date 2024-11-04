import logging
import sys
import os
import json

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
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

vector_store_info = VectorStoreInfo(
    content_info="brief description of visa application",
    metadata_info=[
        MetadataInfo(
            name="document_title",
            type="str",
            description="The title of the document."
        ),
        MetadataInfo(
            name="prev_section_summary",
            type="str",
            description="Summary of the previous section in the document."
        ),
        MetadataInfo(
            name="section_summary",
            type="str",
            description="Summary of the current section in the document."
        ),
        MetadataInfo(
            name="excerpt_keywords",
            type="str",
            description="Keywords extracted from the document excerpt."
        ),
    ],
)

retriever = VectorIndexAutoRetriever(
  index,
  vector_store_info=vector_store_info,
  llm=llm,
  embed_model=embed_model,
  # similarity_top_k=5,
  # verbose=True,
)

def pretty_print_docs(docs):
    formatted_docs = []
    for i, d in enumerate(docs):
        doc_json = d.to_json()
        doc_dict = json.loads(doc_json)

        node = doc_dict.get('node', {})

        text = node.get('text', 'No text available')
        metadata = node.get('metadata', {})

        formatted_metadata = json.dumps(metadata, indent=2)

        formatted_docs.append(
            f"Document {i+1}:\n\nText:\n\n{text}\n\nMetadata:\n\n{formatted_metadata}"
        )
    
    print(f"\n{'-' * 100}\n".join(formatted_docs))

query = input("Enter query: ")

pretty_print_docs(retriever.retrieve(query))
