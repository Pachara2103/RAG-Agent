import streamlit as st
from src.rag import get_pipeline
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="RAG System", page_icon="⚙️", layout="centered")
st.html(
    """<style>.stAppDeployButton {display:none;}#MainMenu {visibility: hidden;}</style>"""
)

with st.sidebar:
    st.header("Controls")
    
    if st.button("🗑️ Clear Chat History", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    
    with st.expander("**View Players**", expanded=False):
        st.write("Current Database includes:")
        st.write("- **Stephen Curry**")
        st.write("- **LeBron James** ")
        st.write("- **Luka Doncic** ")
        st.write("- **Klay Thompson** ")
        st.write("- **Nikola Jokic** ")

    st.markdown("---")

    st.subheader("System Info")
    st.write(f"**LLM:** `Llama-3.2 (Ollama)`")
    st.write(f"**Vector DB:** `MongoDB`")
    
    st.markdown("---")
    st.caption("RAG System v1.0 - Powered by LangGraph & LangChain")

if "messages" not in st.session_state:
    st.session_state.messages = []

rag = get_pipeline()

st.title("⚙️RAG System")
st.caption(
    "Ask me anything about Stephen Curry, Lebron James, Luka Doncic, Klay Thompson and Nikola Jokic"
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        content = "ขออภัยครับ ไม่สามารถสร้างคำตอบได้ในขณะนี้"
        with status_placeholder:
            with st.spinner("Thinking ..."):
                response = rag.invoke({
                        "question": user_input,
                        "messages": [HumanMessage(content=user_input)],
                })
                if response['messages']:
                    last_msg = response["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        content = last_msg.content
                
        status_placeholder.empty()
        message_placeholder.markdown(content)
        st.session_state.messages.append({"role": "assistant", "content": content})
