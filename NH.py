import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="NHIF RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 NHIF RAG Chatbot")
st.caption("Ask questions about NHIF services, coverage, benefits, and claims")

PDF_PATH = "nhif.pdf"
PERSIST_DIR = "./nhif_chroma_db"

@st.cache_resource
def load_rag_chain():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    texts = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100).split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    contextualize_q = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and question, create standalone question."), 
        MessagesPlaceholder("chat_history"), ("human", "{input}")
    ])
    
    answer_prompt = ChatPromptTemplate.from_template("""
You are NHIF expert. Use ONLY this context:
{context}
Question: {input}
Answer only from context or say "No info in NHIF docs."
""")
    
    retriever_chain = create_history_aware_retriever(llm, retriever, contextualize_q)
    qa_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(retriever_chain, qa_chain)
    return rag_chain

rag_chain = load_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    
store = {"nhif": st.session_state.chat_history}
def get_session_history(session_id): return store[session_id]

chain = RunnableWithMessageHistory(
    rag_chain, get_session_history, input_messages_key="input", 
    history_messages_key="chat_history", output_messages_key="answer"
)

for msg in st.session_state.chat_history.messages:
    st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

if user_input := st.chat_input("Ask about NHIF..."):
    st.chat_message("user").write(user_input)
    with st.spinner("Thinking..."):
        result = chain.invoke({"input": user_input}, config={"configurable": {"session_id": "nhif"}})
        st.chat_message("assistant").write(result["answer"])

with st.sidebar:
    st.header("ℹ️ About")
    if st.button("🧹 Clear"): st.session_state.chat_history.clear(); st.rerun()
