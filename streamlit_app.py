import os
import time

import requests
import streamlit as st


API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAG System", layout="wide")
st.title("RAG Question Answering (Local FAISS + SentenceTransformers)")

with st.sidebar:
    st.subheader("Upload")
    up = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if up is not None:
        if st.button("Upload & ingest"):
            files = {"file": (up.name, up.getvalue())}
            r = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
            st.write(r.status_code, r.json())

st.subheader("Ask")
question = st.text_input("Question", placeholder="Ask something grounded in your uploaded documents…")
doc_ids = st.text_input("Document IDs (comma-separated, optional)", value="")

if st.button("Ask"):
    payload = {"question": question}
    if doc_ids.strip():
        payload["document_ids"] = [d.strip() for d in doc_ids.split(",") if d.strip()]
    t0 = time.time()
    r = requests.post(f"{API_BASE}/ask", json=payload, timeout=120)
    dt = (time.time() - t0) * 1000
    st.caption(f"HTTP {r.status_code} • {dt:.0f} ms")
    data = r.json()
    st.markdown("### Answer")
    st.write(data.get("answer"))
    st.markdown("### Sources")
    for s in data.get("sources", []):
        st.markdown(f"**{s['document_id']} • {s['chunk_id']} • score={s['score']:.3f}**")
        st.markdown(s.get("highlighted_text") or s.get("text"))

