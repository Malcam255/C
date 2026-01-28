import streamlit as st
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# -----------------------------------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="NHIF RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 NHIF RAG Chatbot")
st.caption("Ask questions about NHIF services, coverage, benefits, and claims")

# -----------------------------------------------------------------------------
# PATHS (EDIT IF NEEDED)
# -----------------------------------------------------------------------------
PDF_PATH = r"C:\Users\Mekzedeck\Desktop\ChatBot\nhif.pdf"
MODEL_PATH = r"C:\Users\Mekzedeck\AppData\Local\nomic.ai\GPT4All\Llama-3.2-3B-Instruct-Q4_0.gguf"
PERSIST_DIR = "./nhif_chroma_db"

# -----------------------------------------------------------------------------
# LOAD & CACHE EVERYTHING (IMPORTANT FOR STREAMLIT)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_rag_chain():
    # Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    # Embeddings + Vector DB
    embeddings = GPT4AllEmbeddings()

    vectorstore = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIR
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.2, "k": 4}
    )

    # LLM
    llm = GPT4All(
        model=MODEL_PATH,
        n_threads=8
    )

    # Question contextualizer
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Given a chat history and the latest user question,
create a standalone question that incorporates context from the chat history."""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Answer prompt
    answer_prompt = ChatPromptTemplate.from_template("""
You are an NHIF expert assistant. Use ONLY the context below.

CONTEXT:
{context}

QUESTION: {input}

RULES:
- Answer only from NHIF context
- If missing, say: "I don't have that information in the NHIF documents."
- Be concise and factual

ANSWER:
""")

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_chain = create_stuff_documents_chain(llm, answer_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain

rag_chain = load_rag_chain()

# -----------------------------------------------------------------------------
# CHAT HISTORY (SESSION BASED)
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
    output_messages_key="answer",
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
            st.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This chatbot answers questions strictly from NHIF PDF documents using local GPT4All + LangChain RAG.")

    if st.button("🧹 Clear chat"):
        st.session_state.chat_history.clear()
        st.rerun()