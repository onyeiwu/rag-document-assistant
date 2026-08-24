# ============================================================
# app.py — RAG Document Assistant
# ============================================================

import streamlit as st
import faiss
import pickle
import os
import re
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from groq import Groq
from dotenv import load_dotenv

# ── Load environment variables ───────────────────────────────
load_dotenv()

# ── Works both locally (.env) and on Streamlit Cloud (secrets) ──
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

# ── Set up Groq client ───────────────────────────────────────
client = Groq(api_key=api_key)

# ── Load embedding model ─────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ── Streamlit page configuration ─────────────────────────────
st.set_page_config(
    page_title="📚 RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# ============================================================
# Helper — Auto Model Fallback
# ============================================================

def call_groq(messages, temperature=0.1, max_tokens=512):
    models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "groq/compound-mini",
        "qwen/qwen3.6-27b"
    ]

    last_error = None
    for model in models:
        try:
            response = client.chat.completions.create(
                model       = model,
                messages    = messages,
                temperature = temperature,
                max_tokens  = max_tokens
            )
            return response
        except Exception as e:
            if "rate_limit" in str(e) or "429" in str(e) or "413" in str(e):
                last_error = e
                continue
            else:
                raise e

    raise Exception(f"All models hit rate limit — please wait and try again. Last error: {last_error}")

# ============================================================
# PART 2 — PDF Processing Functions
# ============================================================

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page": i + 1,
            "text": text
        })
    return pages

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def is_useful_page(text):
    text_lower = text.lower()
    words      = text.split()

    if len(words) < 30:
        return False
    if text_lower.count("references") > 2:
        return False
    if text_lower.count("contents") > 2:
        return False
    if text_lower.count("index") > 3:
        return False
    if "all rights reserved" in text_lower:
        return False

    number_count = len(re.findall(r'\b\d+\b', text))
    if number_count > 50:
        return False

    return True

def process_pages(pages):
    cleaned_pages = []
    for page in pages:
        cleaned = clean_text(page["text"])
        if is_useful_page(cleaned):
            cleaned_pages.append({
                "page": page["page"],
                "text": cleaned
            })
    return cleaned_pages

def chunk_text(pages, chunk_size=300, overlap=50):
    chunks        = []
    all_words     = []
    word_page_map = []

    for page in pages:
        words = page["text"].split()
        all_words.extend(words)
        word_page_map.extend([page["page"]] * len(words))

    for i in range(0, len(all_words), chunk_size - overlap):
        chunk_words = all_words[i : i + chunk_size]
        chunk_text  = " ".join(chunk_words)
        page_num    = word_page_map[i] if i < len(word_page_map) else word_page_map[-1]

        if chunk_text.strip():
            chunks.append({
                "page"     : page_num,
                "chunk_id" : len(chunks) + 1,
                "text"     : chunk_text
            })

    return chunks

def create_faiss_index(chunks):
    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=False,
        batch_size=32
    )

    embeddings_np         = np.array(embeddings).astype('float32')
    embeddings_normalized = normalize(embeddings_np, norm='l2')

    dimension = embeddings_normalized.shape[1]
    index     = faiss.IndexFlatIP(dimension)
    index.add(embeddings_normalized)

    return index, chunks

# ============================================================
# PART 3 — Search and Answer Functions
# ============================================================

def search_chunks(question, index, chunks, k=6):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')
    question_embedding = normalize(question_embedding, norm='l2')

    scores, indices = index.search(question_embedding, k * 3)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        text  = chunk["text"]

        if len(text.split()) < 30:
            continue
        if text.count(".") > 15 and len(text.split()) < 100:
            continue

        results.append({
            "chunk_id" : chunk["chunk_id"],
            "page"     : chunk["page"],
            "score"    : scores[0][i],
            "text"     : chunk["text"]
        })

        if len(results) == k:
            break

    return results


def ask_rag(question, index, chunks, conversation_history, k=5):
    results = search_chunks(question, index, chunks, k=k)

    if not results:
        return "Sorry, I could not find any relevant information in the document.", [], []

    context = ""
    sources = []

    for i, result in enumerate(results):
        context += f"\n--- Context {i+1} (Page {result['page']}) ---\n"
        context += result["text"]
        context += "\n"
        sources.append(result["page"])

    prompt = f"""Answer the question below using ONLY the provided context.
Be concise — answer in 3 to 5 sentences maximum.
Always mention which page the information came from.
If the answer is not in the context say "I could not find this information in the provided context."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (3-5 sentences only):"""

    messages = [
        {
            "role"   : "system",
            "content": "You are a concise assistant. Answer in 3-5 sentences only using the provided context. Give the final answer directly with no extra explanation."
        }
    ]

    messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})

    response     = call_groq(messages, temperature=0.1, max_tokens=512)
    raw_answer   = response.choices[0].message.content
    clean_answer = re.sub(r'<think>[\s\S]*?</think>', '', raw_answer)
    clean_answer = re.sub(r'</?think>', '', clean_answer)
    clean_answer = re.sub(r'\n{3,}', '\n\n', clean_answer)
    clean_answer = clean_answer.strip()

    conversation_history.append({
        "role"   : "user",
        "content": f"QUESTION: {question}"
    })
    conversation_history.append({
        "role"   : "assistant",
        "content": clean_answer
    })

    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

    return clean_answer, sources, results

# ============================================================
# FEATURE 1 — Document Summary (Fixed — all 5 points)
# ============================================================

def generate_document_summary(chunks, num_chunks=10):
    sample_chunks = chunks[:num_chunks]

    context = ""
    for i, chunk in enumerate(sample_chunks):
        context += f"\n--- Section {i+1} (Page {chunk['page']}) ---\n"
        context += chunk["text"]
        context += "\n"

    prompt = f"""You MUST provide ALL 4 sections below. Do not skip any section.
Use the document text provided to fill each section.

DOCUMENT TEXT:
{context}

Now write ALL 5 sections. Do not skip any:

**1. Document Title/Topic**
[Write what this document is about]

**2. Main Themes**
[List 3 to 5 main themes covered in this document]

**3. Key Concepts**
[List 5 important concepts or terms mentioned]

**4. Who Is This For**
[Describe who would benefit from reading this document]"""

    messages = [
        {
            "role"   : "system",
            "content": "You are a document summarizer. You MUST write all 5 sections completely. Never skip a section. Always use the provided document text as your source."
        },
        {
            "role"   : "user",
            "content": prompt
        }
    ]

    response      = call_groq(messages, temperature=0.1, max_tokens=800)
    raw_summary   = response.choices[0].message.content
    clean_summary = re.sub(r'<think>[\s\S]*?</think>', '', raw_summary)
    clean_summary = re.sub(r'</?think>', '', clean_summary)
    clean_summary = clean_summary.strip()

    return clean_summary

# ============================================================
# FEATURE 2 — Question Suggestions (Fixed)
# ============================================================

def generate_question_suggestions(chunks, num_chunks=10):
    if len(chunks) <= num_chunks:
        sample_chunks = chunks
    else:
        step          = len(chunks) // num_chunks
        sample_chunks = [chunks[i] for i in range(0, len(chunks), step)]
        sample_chunks = sample_chunks[:num_chunks]

    # Build short context from each chunk
    context = ""
    for i, chunk in enumerate(sample_chunks):
        words      = chunk["text"].split()[:80]
        short_text = " ".join(words)
        context   += f"[Page {chunk['page']}]: {short_text}\n\n"

    prompt = f"""Here is text from a document. Generate 5 questions about THIS specific document.

{context}

Write exactly 5 questions about the text above.
Each question must start with a number and period like: 1. 2. 3. 4. 5.
Questions must be specific to THIS document only.

1."""

    messages = [
        {
            "role"   : "system",
            "content": "Generate exactly 5 numbered questions specific to the document text provided. Do not use generic questions. Each question must relate directly to the content shown."
        },
        {
            "role"   : "user",
            "content": prompt
        }
    ]

    try:
        response          = call_groq(messages, temperature=0.4, max_tokens=250)
        raw               = response.choices[0].message.content
        clean             = re.sub(r'<think>[\s\S]*?</think>', '', raw)
        clean             = re.sub(r'</?think>', '', clean)
        clean             = clean.strip()

        # Add back "1." since we used it as prompt suffix
        full_text = "1." + clean if not re.match(r'^\d', clean) else clean

        # Parse all numbered lines
        questions = []
        for line in full_text.split('\n'):
            line  = line.strip()
            match = re.match(r'^\d+[\.\)]\s*(.+)', line)
            if match:
                q = match.group(1).strip()
                if len(q) > 15:
                    questions.append(q)

        if len(questions) >= 3:
            return questions[:5]
        else:
            raise Exception("Not enough questions parsed")

    except Exception as e:
        # Build document specific fallback from actual content
        first_words  = " ".join(chunks[0]["text"].split()[:20])
        middle_words = " ".join(chunks[len(chunks)//2]["text"].split()[:20])
        last_words   = " ".join(chunks[-1]["text"].split()[:20])

        return [
            f"What is the main purpose of this document?",
            f"What does this text say about: {first_words[:50]}?",
            f"Can you explain the concept mentioned here: {middle_words[:50]}?",
            f"What conclusions are drawn in this document?",
            f"What does this document say about: {last_words[:50]}?"
        ]

# ============================================================
# FEATURE 4 — Download Chat History
# ============================================================

def generate_chat_download(chat_messages, uploaded_files_names):
    content  = ""
    content += "=" * 60 + "\n"
    content += "RAG DOCUMENT ASSISTANT — CHAT HISTORY\n"
    content += "=" * 60 + "\n\n"

    content += "DOCUMENTS USED:\n"
    for name in uploaded_files_names:
        content += f"   - {name}\n"
    content += "\n"
    content += "=" * 60 + "\n\n"

    for message in chat_messages:
        if message["role"] == "user":
            content += f"YOU:\n"
            content += f"   {message['content']}\n\n"
        else:
            content += f"ASSISTANT:\n"
            content += f"   {message['content']}\n"
            if "sources" in message:
                content += f"   Sources — Pages: {message['sources']}\n"
            if "confidence" in message:
                content += f"   Confidence: {message['confidence_label']} ({message['confidence']}%)\n"
            content += "\n"
        content += "-" * 60 + "\n\n"

    return content

# ============================================================
# FEATURE 5 — Confidence Score (Fixed — 2 decimal places)
# ============================================================

def calculate_confidence(results):
    if not results:
        return 0.00, "🔴 Low"

    scores     = [result["score"] for result in results]
    avg_score  = sum(scores) / len(scores)

    # ── Fixed to exactly 2 decimal places ────────────────────
    confidence = round(avg_score * 100, 2)

    if confidence >= 70:
        label = "🟢 High"
    elif confidence >= 50:
        label = "🟡 Medium"
    else:
        label = "🔴 Low"

    return confidence, label

# ============================================================
# PART 4 — Main Streamlit Interface
# ============================================================

def main():

    # ── Session state initialization ─────────────────────────
    if "index" not in st.session_state:
        st.session_state.index = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "uploaded_files_names" not in st.session_state:
        st.session_state.uploaded_files_names = []
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False
    if "document_summary" not in st.session_state:
        st.session_state.document_summary = None
    if "question_suggestions" not in st.session_state:
        st.session_state.question_suggestions = []

    # ── App Title ─────────────────────────────────────────────
    st.title("📚 RAG Document Assistant")
    st.markdown("Upload one or more PDF documents and ask questions about them.")
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.header("📁 Document Management")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can upload one or more PDF files"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            for f in uploaded_files:
                st.write(f"📄 {f.name}")

            if st.button("🔄 Process Documents", use_container_width=True):
                with st.spinner("Processing documents... please wait..."):
                    all_pages = []

                    for pdf_file in uploaded_files:
                        pages = extract_text_from_pdf(pdf_file)
                        all_pages.extend(pages)

                    cleaned_pages = process_pages(all_pages)
                    chunks        = chunk_text(cleaned_pages)
                    index, chunks = create_faiss_index(chunks)

                    st.session_state.index                = index
                    st.session_state.chunks               = chunks
                    st.session_state.conversation_history = []
                    st.session_state.chat_messages        = []
                    st.session_state.uploaded_files_names = [f.name for f in uploaded_files]
                    st.session_state.processing_done      = True
                    st.session_state.document_summary     = generate_document_summary(chunks)
                    st.session_state.question_suggestions = generate_question_suggestions(chunks)

                st.success("✅ Documents processed successfully!")
                st.rerun()

        st.divider()

        if st.session_state.uploaded_files_names:
            st.subheader("📋 Loaded Documents:")
            for name in st.session_state.uploaded_files_names:
                st.write(f"📄 {name}")

        st.divider()

        if st.session_state.processing_done:
            if st.button("🗑️ Remove All Documents", use_container_width=True):
                st.session_state.index                = None
                st.session_state.chunks               = None
                st.session_state.conversation_history = []
                st.session_state.chat_messages        = []
                st.session_state.uploaded_files_names = []
                st.session_state.processing_done      = False
                st.session_state.document_summary     = None
                st.session_state.question_suggestions = []
                st.success("✅ All documents removed!")
                st.rerun()

        # ── Clear and Download buttons ────────────────────────
        if st.session_state.chat_messages:
            st.divider()

            if st.button("🧹 Clear Chat History", use_container_width=True):
                st.session_state.conversation_history = []
                st.session_state.chat_messages        = []
                st.rerun()

            chat_content = generate_chat_download(
                st.session_state.chat_messages,
                st.session_state.uploaded_files_names
            )
            st.download_button(
                label               = "⬇️ Download Chat History",
                data                = chat_content,
                file_name           = "chat_history.txt",
                mime                = "text/plain",
                use_container_width = True
            )

    # ── Main Chat Area ────────────────────────────────────────
    if not st.session_state.processing_done:
        st.info("👈 Please upload a PDF document from the sidebar to get started.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📤 Step 1")
            st.markdown("Upload one or more PDF files from the sidebar")
        with col2:
            st.markdown("### 🔄 Step 2")
            st.markdown("Click **Process Documents** to build the index")
        with col3:
            st.markdown("### 💬 Step 3")
            st.markdown("Ask any question about your documents")

    else:
        # ── Show document info ────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", len(st.session_state.chunks))
        with col2:
            st.metric("Documents Loaded", len(st.session_state.uploaded_files_names))

        st.divider()

        # ── Show document summary ─────────────────────────────
        if st.session_state.document_summary:
            with st.expander("📋 Document Summary — Click to expand", expanded=True):
                st.markdown(st.session_state.document_summary)

        st.divider()

        # ── Show question suggestions ─────────────────────────
        if st.session_state.question_suggestions:
            with st.expander("💡 Suggested Questions — Click any to ask", expanded=True):
                st.markdown("**Click any question below:**")
                for i, suggestion in enumerate(st.session_state.question_suggestions):
                    if st.button(
                        f"{suggestion} ❓",
                        key=f"suggestion_{i}",
                        use_container_width=True
                    ):
                        st.session_state.chat_messages.append({
                            "role"   : "user",
                            "content": suggestion
                        })
                        answer, sources, results = ask_rag(
                            suggestion,
                            st.session_state.index,
                            st.session_state.chunks,
                            st.session_state.conversation_history
                        )
                        confidence, label = calculate_confidence(results)
                        st.session_state.chat_messages.append({
                            "role"             : "assistant",
                            "content"          : answer,
                            "sources"          : sources,
                            "confidence"       : confidence,
                            "confidence_label" : label
                        })
                        st.rerun()

        st.divider()

        # ── Display chat history ──────────────────────────────
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    if "confidence" in message:
                        st.caption(
                            f"📄 Sources — Pages: {message['sources']}   |   "
                            f"🎯 Confidence: {message['confidence_label']} "
                            f"({message['confidence']:.2f}%)"
                        )
                    else:
                        st.caption(f"📄 Sources — Pages: {message['sources']}")

        # ── Chat input ────────────────────────────────────────
        question = st.chat_input("Ask a question about your documents...")

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            st.session_state.chat_messages.append({
                "role"   : "user",
                "content": question
            })

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources, results = ask_rag(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.conversation_history
                    )
                confidence, label = calculate_confidence(results)
                st.markdown(answer)
                st.caption(
                    f"📄 Sources — Pages: {sources}   |   "
                    f"🎯 Confidence: {label} ({confidence:.2f}%)"
                )

            st.session_state.chat_messages.append({
                "role"             : "assistant",
                "content"          : answer,
                "sources"          : sources,
                "confidence"       : confidence,
                "confidence_label" : label
            })

            st.rerun()

# ── Run the app ───────────────────────────────────────────────
if __name__ == "__main__":
    main()