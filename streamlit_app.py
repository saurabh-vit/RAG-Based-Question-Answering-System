import os
import time

import requests
import streamlit as st


API_BASE = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8000").rstrip("/")


def post_to_api(path: str, **kwargs):
    """Call the API and show a useful message if the backend is offline."""
    try:
        return requests.post(f"{API_BASE}{path}", **kwargs)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        st.error(
            f"The RAG API did not respond at {API_BASE}. "
            "Ensure the FastAPI server is running and check its terminal for errors."
        )
        return None

st.set_page_config(page_title="RAG System", layout="wide")
st.title("RAG Question Answering (Local FAISS + SentenceTransformers)")

with st.sidebar:
    st.subheader("Upload")
    up = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if up is not None:
        if st.button("Upload & ingest"):
            files = {"file": (up.name, up.getvalue())}
            r = post_to_api("/upload", files=files, timeout=60)
            if r is not None:
                if r.ok:
                    st.write(r.status_code, r.json())
                else:
                    st.error(f"Upload failed ({r.status_code}): {r.text}")

st.subheader("Ask")
question = st.text_input("Question", placeholder="Ask something grounded in your uploaded documents…")
doc_ids = st.text_input("Document IDs (comma-separated, optional)", value="")

if st.button("Ask"):
    payload = {"question": question}
    if doc_ids.strip():
        payload["document_ids"] = [d.strip() for d in doc_ids.split(",") if d.strip()]
    t0 = time.time()
    r = post_to_api("/ask", json=payload, timeout=120)
    if r is None:
        st.stop()
    if not r.ok:
        st.error(f"Question failed ({r.status_code}): {r.text}")
        st.stop()
    dt = (time.time() - t0) * 1000
    st.caption(f"HTTP {r.status_code} • {dt:.0f} ms")
    data = r.json()
    st.markdown("### Answer")
    st.write(data.get("answer"))
    st.markdown("### Sources")
    for s in data.get("sources", []):
        st.markdown(f"**{s['document_id']} • {s['chunk_id']} • score={s['score']:.3f}**")
        st.markdown(s.get("highlighted_text") or s.get("text"))

