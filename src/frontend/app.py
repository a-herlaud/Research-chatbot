import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Research Chatbot", page_icon="💬")
st.title("Research Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} retrieved chunks)"):
        for source in sources:
            st.markdown(
                f"- **{source['title']}** "
                f"(`{source['paper_id']}`, score {source['score']})"
            )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/chat", json={"message": prompt}, timeout=30
            )
            response.raise_for_status()
            payload = response.json()
            reply = payload["reply"]
            sources = payload.get("sources", [])
        except requests.RequestException as exc:
            reply = f"Error contacting backend: {exc}"
            sources = []

        st.markdown(reply)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
