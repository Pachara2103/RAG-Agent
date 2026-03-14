import streamlit as st
from src.graph.workflow import get_workflow
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="RAG Agent", page_icon="🧠", layout="centered")
st.html(
    """<style>.stAppDeployButton {display:none;}#MainMenu {visibility: hidden;}</style>"""
)

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Clear Chat History", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        **RAG Agent** Retrieval-Augmented Generation with:
        - Self-Correction
        - Query Rewriting
        - Web Search Fallback
        """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

agent = get_workflow()

st.title("🧠RAG Agent")
st.caption("Ask me anything about your documents. I'll verify and correct my answers.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.status("Running Workflow...", expanded=True) as status:
            st.write("🔍 Analyzing query...")

            user_message = HumanMessage(content=prompt)
            response = agent.invoke({"messages": [user_message]})

            last_message = response["messages"][-1]
            content = last_message.content

            if isinstance(content, list):
                content = "".join(
                    [
                        item.get("text", "")
                        for item in content
                        if item.get("type") == "text"
                    ]
                )

            status.update(label="Complete!", state="complete", expanded=False)

        message_placeholder.markdown(content)

        documents = response.get("filtered_documents")
        if documents:
            with st.expander("📚 Referenced Sources"):
                for i, doc in enumerate(documents):
                    if isinstance(doc, str):
                        st.markdown(f"**{i+1}. Source: **Database**")
                        st.caption(doc[:300] + "...") 
                    else:
                        source = ""
                        if hasattr(doc, "metadata") and doc.metadata:
                            source = doc.metadata.get("source") or doc.metadata.get("url") 

                        st.markdown(f"**{i+1}. Source:** {source}")
                        content = getattr(doc, "page_content", str(doc))
                        st.caption(content[:300] + "...")
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": content})
