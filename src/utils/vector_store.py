from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)

DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",  # Similarity function [-1,1] (close to 1 mean very similar)
    text_key="text",
)

vector_store.create_vector_search_index(dimensions=1024)
