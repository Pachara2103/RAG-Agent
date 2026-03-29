from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
import streamlit as st
from src.vector import vector_store, rerank_model

@st.cache_resource
def get_llm():
    llm = ChatOllama(
      model="llama3.2",
    temperature=0
    )
    return llm

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]

def get_documents(question, threshold=0.6):
    results = vector_store.similarity_search_with_score(question, k=10)
    if not results:
        return []
    return [doc for (doc, score) in results if score>=threshold]

def rerank_documents(query, documents, limit=3):
    if not documents:
        return []

    sentence_pairs = [[query, doc.page_content] for doc in documents]
    scores = rerank_model.predict(sentence_pairs) 

    for i, score in enumerate(scores):
        documents[i].metadata["rerank_score"] = score

    reranked_docs = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)
    return reranked_docs[:limit]


def retrieve_documents_node(state: RAGState) -> RAGState:
    print("\nCalling retriever...")
    documents = get_documents(state["question"])
    return {"documents": documents}

def rerank_document_node(state: RAGState) -> RAGState:
    print("Re-ranking documents...")
    docs = state["documents"]
    reranked_docs = rerank_documents(state["question"],  state["documents"])
    return {"documents": reranked_docs}


def answer_generator_node(state: RAGState) -> RAGState:
    print("Generating answer...")
    references = "\n\n".join([f"ข้อมูลอ้างอิงที่ {i+1}:\n{doc.page_content}" for i ,doc in enumerate(state["documents"])])

    system_propmt = """
     คุณคือผู้ช่วยตอบคำถามภาษาไทยที่มีหน้าที่สรุปข้อมูลอ้างอิงอย่างกระชับ
     จงตอบคำถามโดยใช้ข้อมูลจาก "ข้อมูลอ้างอิง" ที่ให้มา

     คำแนะนำการตอบ:
     1. ตอบให้ตรงประเด็น "เข้าเนื้อหาทันที" ไม่ต้องเกริ่นว่ามีหรือไม่มีข้อมูลในส่วนไหน
     2. หากข้อมูลในอ้างอิงเพียงพอต่อการตอบ ให้ตอบเฉพาะคำตอบนั้นๆ
     3. ไม่ต้องอธิบายที่มาหรือข้อจำกัดของข้อมูลอ้างอิง ยกเว้นแต่ข้อมูลไม่เพียงพอจริงๆ จึงตอบว่าไม่ทราบ
     4. ใช้ภาษาสุภาพและเป็นทางการ

     ข้อมูลอ้างอิง:
     {references}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_propmt),
        ("human", "{question}")
    ])
    rag = prompt | get_llm()

    response = rag.invoke({
        "references": references,
        "question": state["question"]
    })

    return {"messages": [response]}

def get_pipeline():
    graph = StateGraph(RAGState)

    graph.add_node("retriever", retrieve_documents_node)
    graph.add_node("rerank", rerank_document_node)
    graph.add_node("answer generator", answer_generator_node)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "rerank")
    graph.add_edge("rerank", "answer generator")
    graph.add_edge("answer generator", END)

    return graph.compile()
