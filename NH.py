import os
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
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="NHIF RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 NHIF RAG Chatbot")
st.caption("Ask questions about NHIF services, benefits, claims, and coverage")

# ============================================================
# PATHS (EDIT IF DEPLOYING)
# ============================================================
PDF_PATH = "nhif.pdf"   # put pdf in same folder
VECTOR_DB_DIR = "./nhif_chroma_db"
MODEL_PATH = r"C:\Users\Mekzedeck\AppData\Local\nomic.ai\GPT4All\Llama-3.2-3B-Instruct-Q4_0.gguf"

# ============================================================
# LOAD & CACHE RESOURCES
# ============================================================
@st.cache_resource(show_spinner=True)
def load_rag_pipeline():

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)

    embeddings = GPT4AllEmbeddings()

    vectorstore = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.1, "k": 4}
    )

    llm = GPT4All(
        model=MODEL_PATH,
        n_threads=8
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question, "
         "create a standalone question that incorporates context."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    answer_prompt = ChatPromptTemplate.from_template("""
You are an NHIF expert assistant.

CONTEXT:
{context}

QUESTION: {input}

Rules:
- Answer ONLY using the NHIF context
- If missing info say: "I don't have that information in the NHIF documents."
- Be concise and accurate

ANSWER:
""")

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(
        llm, answer_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_chain


rag_chain = load_rag_pipeline()

# ============================================================
# CHAT UI
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about NHIF...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching NHIF documents..."):
            try:
                result = rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "nhif_streamlit"}}
                )
                answer = result["answer"]
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
