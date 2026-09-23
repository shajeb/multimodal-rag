"""Trusted local Streamlit interface. Use the authenticated API for tenant access."""
import streamlit as st
from rag.config import Settings
from rag.service import RAGService

st.set_page_config(page_title="Multimodal RAG", page_icon="📚", layout="wide")
st.title("Multimodal document workspace")
st.caption("Ask questions across PDF text, tables and images, with page-level evidence.")

@st.cache_resource
def get_service(settings, tenant):
    return RAGService(settings, tenant)

with st.sidebar:
    st.header("Workspace")
    tenant = st.text_input("Local tenant", value="local", help="Local administrator selector, not an authentication control.")
    session = st.text_input("Conversation", value="default")
    try:
        settings = Settings.load()
        service = get_service(settings, tenant)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    st.caption(f"Mode: {settings.mode} · Store: {settings.backend}")
    if settings.mode == "demo":
        st.info("Offline demo: lexical retrieval and quoted excerpts. Enable live mode for semantic models and image understanding.")
    with st.form("upload"):
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        submit = st.form_submit_button("Index documents", type="primary")
    if submit:
        for upload in uploads:
            try:
                with st.spinner(f"Indexing {upload.name}"):
                    result = service.ingest_bytes(upload.getvalue(), upload.name)
                st.success(f"{upload.name}: {result['status']} ({result['chunks']} chunks)")
            except Exception as exc:
                st.error(f"{upload.name}: {exc}")
    with st.expander("Long-term memory"):
        note = st.text_area("Save context for this conversation", placeholder="I prefer short answers with units.")
        if st.button("Remember"):
            try:
                service.remember(session, note)
                st.success("Saved")
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Clear this conversation"):
            try:
                service.catalog.clear_memory(session)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

try:
    records = service.catalog.documents()
    unique = {r["document"]: r["source"] for r in records}
    left, right = st.columns([3, 1])
    with left:
        selected = st.selectbox("Search documents", ["All documents"] + list(unique),
                                format_func=lambda x: unique.get(x, x))
    with right:
        modality = st.selectbox("Content", ["All types", "text", "table", "image"])
    with st.expander(f"Document versions ({len(records)})"):
        st.dataframe(records, use_container_width=True)
    history = service.catalog.history(session)
    for turn in history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
except ValueError as exc:
    st.error(str(exc))
    st.stop()

question = st.chat_input("Ask about your PDFs")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer, sources = "", []
        try:
            for event in service.stream(question, session, None if selected == "All documents" else selected,
                                        None if modality == "All types" else modality):
                if event["event"] == "token":
                    answer += event["text"]
                    placeholder.markdown(answer)
                elif event["event"] == "sources":
                    sources = event["sources"]
                elif event["event"] == "done":
                    for warning in event["warnings"]:
                        st.warning(warning)
            with st.expander("Evidence"):
                assets = {c.id: c.asset for c, _ in service.catalog.active() if c.asset}
                for source in sources:
                    st.markdown(f"**[{source['label']}] {source['source']} — page {source['page']} · {source['kind']}**")
                    st.text(source["text"])
                    if source["id"] in assets:
                        st.image(assets[source["id"]], caption=f"Page {source['page']} image evidence")
        except Exception as exc:
            st.error(f"Request failed: {exc}")
