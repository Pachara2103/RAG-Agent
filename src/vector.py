from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from sentence_transformers import CrossEncoder
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import streamlit as st

@st.cache_resource
def init_models():
    device = "cpu"
    encode_kwargs = {"normalize_embeddings": True}
    embed = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs=encode_kwargs
    )
    
    rerank = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device)
    
    return embed, rerank

@st.cache_resource
def init_vector_store(_embedding_model):
    load_dotenv()
    URI = os.getenv('MONGODB_URI')
    DB_NAME = os.getenv('DB_NAME')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME')
    INDEX_NAME = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')

    client = MongoClient(URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    v_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=_embedding_model,
        index_name=INDEX_NAME,
        relevance_score_fn="cosine",
        text_key="page_content"
    )
    return v_store

embedding_model, rerank_model = init_models()
vector_store = init_vector_store(embedding_model)