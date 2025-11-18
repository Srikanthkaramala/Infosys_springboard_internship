import os
import io
import json
import time
import tempfile
import streamlit as st
from fpdf import FPDF
from gtts import gTTS
import speech_recognition as sr
from deep_translator import GoogleTranslator

# ---- LangChain modern imports ----
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="📘 Policy RAG Assistant", layout="wide", page_icon="📘")
st.title("📘 Policy RAG Assistant — LM Studio + FAISS")

# LM Studio setup
os.environ["OPENAI_API_BASE"] = "http://localhost:12345/v1"
os.environ["OPENAI_API_KEY"] = "lm-studio"
MODEL_NAME = "tinyllama-1.1b-chat-v1.0"

# Vector DB path
INDEX_PATH = "policy_index"

# Language map
LANG_CODES = {
    "English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta",
    "Spanish": "es", "French": "fr", "German": "de"
}
answer_language = st.sidebar.selectbox("🌐 Answer Language", list(LANG_CODES.keys()))
lang_code = LANG_CODES[answer_language]

# =====================================================
# HELPERS
# =====================================================
def translate_text(text: str, target_lang: str) -> str:
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text

def create_pdf(title: str, text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{title}\n\n{text}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        path = tmp.name
    pdf.output(path)
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data

def tts_audio(text: str, lang: str = "en") -> bytes | None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            path = tmp.name
        gTTS(text=text, lang=lang).save(path)
        with open(path, "rb") as f:
            data = f.read()
        os.remove(path)
        return data
    except Exception:
        return None

# =====================================================
# LOAD OR BUILD VECTORSTORE
# =====================================================
st.info("🔍 Loading FAISS vector database...")

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    st.success(f"✅ Loaded FAISS index from {INDEX_PATH}")
except Exception as e:
    st.error(f"❌ Could not load FAISS index. Run 'build_vectorstore.py' first.\n\n{e}")
    st.stop()

# =====================================================
# LLM + PROMPT + RAG CHAIN
# =====================================================
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.5,
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    request_timeout=90,
)

prompt = ChatPromptTemplate.from_template("""
You are a policy assistant that summarizes and lists relevant policies clearly.
Use only the provided context to answer.
If not found, say: "The information is not available in the provided context."

Format:
Policy 1: <Title or Summary>
Policy 2: <Title or Summary>
...

Context:
{context}

Question:
{question}

Answer:
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =====================================================
# UI — Query Input (Text + Voice)
# =====================================================
st.subheader("💬 Ask a Question")
query = st.text_input("Type your question:")

if st.button("🎙 Speak your question"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎧 Listening (6s)...")
        try:
            audio = r.listen(source, timeout=6, phrase_time_limit=6)
            query = r.recognize_google(audio)
            st.success(f"🗣 You said: {query}")
        except Exception as e:
            st.warning(f"Voice input error: {e}")

# =====================================================
# PROCESS QUERY
# =====================================================
if query:
    st.info("🧠 Generating response...")
    query_en = translate_text(query, "en")

    try:
        answer = rag_chain.invoke(query_en)
    except Exception as e:
        st.error(f"❌ Error generating response: {e}")
        st.stop()

    translated_answer = translate_text(answer, lang_code)
    st.subheader("🧾 Answer")
    st.write(translated_answer)

    audio_data = tts_audio(translated_answer, lang_code)
    if audio_data:
        st.audio(audio_data, format="audio/mp3")

    st.download_button(
        "⬇️ Download PDF",
        data=create_pdf(f"Q: {query}", translated_answer),
        file_name="answer.pdf",
        mime="application/pdf",
    )
    st.download_button(
        "⬇️ Download JSON",
        data=json.dumps({"question": query, "answer": translated_answer}, indent=2, ensure_ascii=False),
        file_name="answer.json",
        mime="application/json",
    )

st.markdown("---")
st.caption("⚡ Powered by TinyLlama + LM Studio + FAISS + LangChain")
