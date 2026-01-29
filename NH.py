import os
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.llms import GPT4All
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# -----------------------------------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NHIF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 NHIF RAG Chatbot")
st.caption("Ask questions strictly from NHIF documents")

# -----------------------------------------------------------------------------
# PATHS (LINUX + UPSUN SAFE)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PERSIST_DIR = os.path.join(BASE_DIR, "nhif_chroma_db")

PDF_PATH = os.path.join(DATA_DIR, "nhif.pdf")

MODEL_NAME = "Llama-3.2-3B-Instruct-Q4_0.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD & CACHE RAG CHAIN
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_rag_chain():

    # ---- Load PDF ----
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # ---- Split text ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # ---- Embeddings (Cloud & VPS safe) ----
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---- Vector DB ----
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=PERSIST_DIR
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.2, "k": 4}
    )

    # ---- Download GPT4All model if missing ----
    if not os.path.exists(MODEL_PATH):
        from gpt4all import GPT4All
        GPT4All.download_model(
            MODEL_NAME,
            model_path=MODEL_DIR
        )

    # ---- LLM ----
    llm = GPT4All(
        model=MODEL_PATH,
        n_threads=8
    )

    # ---- Question contextualizer ----
    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given chat history and the latest question, create a standalone question."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # ---- Answer prompt ----
    answer_prompt = ChatPromptTemplate.from_template("""
You are an NHIF expert assistant.
Use ONLY the context provided.

CONTEXT:
{context}

QUESTION:
{input}

RULES:
- Answer strictly from the NHIF context
- If information is missing, say:
  "I don't have that information in the NHIF documents."
- Be concise and factual

ANSWER:
""")

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt
    )

    qa_chain = create_stuff_documents_chain(
        llm,
        answer_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )

    return rag_chain

rag_chain = load_rag_chain()

# -----------------------------------------------------------------------------
# CHAT HISTORY
# -----------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

store = {"nhif": st.session_state.chat_history}

def get_session_history(session_id: str):
    return store[session_id]

conversational_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# -----------------------------------------------------------------------------
# CHAT UI
# -----------------------------------------------------------------------------
for msg in st.session_state.chat_history.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Ask a question about NHIF...")

if user_input:
    st.chat_message("user").write(user_input)

    with st.spinner("Searching NHIF documents..."):
        try:
            result = conversational_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "nhif"}}
            )

            answer = result["answer"]
            st.chat_message("assistant").write(answer)

        except Exception as e:
            st.error(str(e))

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This chatbot uses Retrieval-Augmented Generation (RAG) "
        "to answer questions strictly from NHIF PDF documents."
    )

    if st.button("🧹 Clear chat"):
        st.session_state.chat_history.clear()
        st.rerun()
