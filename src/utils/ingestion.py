from vector_store import vector_store
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


file_path = "data/info.txt"

def store_content(file_path):
    reader = TextLoader(file_path=file_path, encoding="utf-8")
    docs = reader.load()

    if not docs:
        print("this file not have content")
        return
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n"]
    )
    chunks = text_splitter.split_documents(docs)
    vector_store.add_documents(chunks)
    print('store success!')
    return

store_content(file_path)