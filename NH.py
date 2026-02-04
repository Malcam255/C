# app.py

import os
import time
import logging
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("NHIF-RAG")

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="NHIF RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 NHIF RAG Chatbot")
st.caption("Ask questions about NHIF services, benefits, claims, and coverage")

# Optional: Show logs in UI
log_box = st.expander("📊 System logs", expanded=True)
log_area = log_box.empty()

def ui_log(msg):
    logger.info(msg)
    log_area.markdown(f"`{msg}`")

# ============================================================
# PATHS
# ============================================================
PDF_PATH = "nhif.pdf"
VECTOR_DB_DIR = "./nhif_chroma_db"
MODEL_PATH = r"C:\Users\Mekzedeck\AppData\Local\nomic.ai\GPT4All\Llama-3.2-1B-Instruct-Q4_0.gguf"

# ============================================================
# VECTORSTORE (LAZY + LOGGED)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_vectorstore():

    start_total = time.perf_counter()
    embeddings = GPT4AllEmbeddings()

    if os.path.exists(VECTOR_DB_DIR):
        start = time.perf_counter()
        vs = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        ui_log(f"[VECTORSTORE] Loaded from disk in {time.perf_counter() - start:.2f}s")
        return vs

    ui_log("[STARTUP] Vectorstore not found – building embeddings")

    start = time.perf_counter()
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    ui_log(f"[PDF] Loaded {len(docs)} pages in {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(600, 100)
    chunks = splitter.split_documents(docs)
    ui_log(f"[CHUNKS] Created {len(chunks)} chunks in {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    ui_log(f"[EMBEDDINGS] Built + saved in {time.perf_counter() - start:.2f}s")

    ui_log(f"[TOTAL] Index built in {time.perf_counter() - start_total:.2f}s")
    return vectorstore

# ============================================================
# RAG PIPELINE 
# ============================================================
def load_rag_pipeline():

    start_total = time.perf_counter()

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.1, "k": 4}
    )

    start = time.perf_counter()
    llm = GPT4All(
        model=MODEL_PATH,
        n_threads=8
    )
    ui_log(f"[LLM] Model loaded in {time.perf_counter() - start:.2f}s")

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, create a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    answer_prompt = ChatPromptTemplate.from_template("""
You are an NHIF expert assistant.

CONTEXT:
{context}

QUESTION: {input}

ANSWER:
""")

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    ui_log(f"[TOTAL] RAG pipeline ready in {time.perf_counter() - start_total:.2f}s")

    store = {}

    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

# ============================================================
# SESSION STATE
# ============================================================
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# CHAT UI
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about NHIF...")

if user_input:

    if st.session_state.rag_chain is None:
        with st.spinner("⚙️ Initializing NHIF AI assistant..."):
            st.session_state.rag_chain = load_rag_pipeline()

    rag_chain = st.session_state.rag_chain

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("🔎 Searching NHIF documents..."):
            result = rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "nhif_streamlit"}}
            )
            answer = result["answer"]

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
