"""
NHIF RAG CHATBOT
Topic-aware conversational chatbot
No question rewriting
Optimized for local GPT4All
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GPT4All

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# =============================================================================
# CONFIG
# =============================================================================

PDF_PATH = r"C:\Users\Mekzedeck\Desktop\ChatBot\nhif.pdf"
VECTOR_DB_DIR = "./nhif_chroma_db"
MODEL_PATH = r"C:\Users\Mekzedeck\AppData\Local\nomic.ai\GPT4All\Llama-3.2-3B-Instruct-Q4_0.gguf"

# =============================================================================
# 1. LOAD & SPLIT DOCUMENTS
# =============================================================================

print("🔄 Loading NHIF documents...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
print(f"✅ Loaded and split into {len(chunks)} chunks")

# =============================================================================
# 2. VECTOR STORE (PERSISTENT)
# =============================================================================

print("🔄 Preparing vector database...")

embeddings = GPT4AllEmbeddings()

if os.path.exists(VECTOR_DB_DIR):
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    print("✅ Existing vector DB loaded")
else:
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print("✅ New vector DB created")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# =============================================================================
# 3. LOAD LLM
# =============================================================================

print("🔄 Loading LLM...")

llm = GPT4All(
    model=MODEL_PATH,
    backend="llama",
    n_threads=8,
    verbose=False
)

print("✅ LLM ready")

# =============================================================================
# 4. PROMPT (STRICT NHIF MODE)
# =============================================================================

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are an NHIF expert assistant.

Use ONLY the information from the NHIF context below.

CONTEXT:
{context}

QUESTION:
{input}

RULES:
- Do NOT guess
- If the answer is not in the context, say:
  "I don't have that information in the NHIF documents."
- Be clear and concise

ANSWER:
""")

qa_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# =============================================================================
# 5. CHAT MEMORY + TOPIC MANAGER
# =============================================================================

chat_store = {}
topic_store = {}

def get_history(session_id: str):
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()
    return chat_store[session_id]

def is_new_topic(text: str) -> bool:
    triggers = [
        "tell me about", "explain", "what is", "describe",
        "information about", "details about"
    ]
    return any(t in text.lower() for t in triggers)

def build_query(session_id: str, user_input: str) -> str:
    if is_new_topic(user_input):
        topic_store[session_id] = user_input
        return user_input

    if session_id in topic_store:
        return topic_store[session_id] + " " + user_input

    return user_input

chatbot = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# =============================================================================
# 6. CHAT LOOP
# =============================================================================

def run_chatbot():
    print("\n" + "=" * 70)
    print("🤖 NHIF CHATBOT")
    print("💡 Topic-aware | No question rewriting | Local AI")
    print("📝 Type 'exit' to quit")
    print("=" * 70)

    session_id = "nhif_session_001"

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        if not user_input:
            print("⚠️ Please ask a question.")
            continue

        query = build_query(session_id, user_input)

        try:
            result = chatbot.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )
            print("\n✅ Assistant:", result["answer"])

        except Exception as e:
            print("❌ Error:", e)

# =============================================================================
# 7. RUN
# =============================================================================

if __name__ == "__main__":
    run_chatbot()
