import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import chromadb

collection_name = "vfs"

db = chromadb.PersistentClient(path="./db")
db.delete_collection(collection_name)