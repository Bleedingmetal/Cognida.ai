import asyncio
import json
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

collection_name = "visa"

db = chromadb.PersistentClient(path="./db")
chroma_collection = db.get_or_create_collection(collection_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

async def get_and_print_nodes():
  nodes = await vector_store.aget_nodes()
  for i in range(0, 2):
    node_json = nodes[i].to_json()
    formatted_json = json.dumps(json.loads(node_json), indent=2)
    print(formatted_json)

asyncio.run(get_and_print_nodes())
