import os
import pickle
from io import BytesIO
import re
import streamlit as st
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import docx2txt
import pandas as pd
from PIL import Image
import pytesseract
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import faiss
from typing import List, Dict
import numpy as np
import json
import sqlite3

# Load environment variables
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
FORM_RECOGNIZER_ENDPOINT = os.getenv("FORM_RECOGNIZER_ENDPOINT")
FORM_RECOGNIZER_KEY = os.getenv("FORM_RECOGNIZER_KEY")
EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
PERSIST_DIR = "faiss_data"
os.makedirs(PERSIST_DIR, exist_ok=True)
DOCS_META_FMT   = os.path.join(PERSIST_DIR, "{}_docs.pkl")
SELECTED_FMT    = os.path.join(PERSIST_DIR, "{}_selected.pkl")
# Initialize clients
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY or not AZURE_API_VERSION:
    raise ValueError("Missing Azure OpenAI settings.")
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
if not FORM_RECOGNIZER_ENDPOINT or not FORM_RECOGNIZER_KEY:
    raise ValueError("Missing Form Recognizer settings.")
doc_client = DocumentAnalysisClient(
    endpoint=FORM_RECOGNIZER_ENDPOINT,
    credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
)
MAX_SIZE_MB = 50

# New: Directory for storing actual uploaded files
UPLOAD_DIR = "persisted_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# New: SQLite Database initialization
DB_PATH = os.path.join(PERSIST_DIR, "faiss_data.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            extracted_text TEXT,
            UNIQUE(folder, filename)
        )
    """)
    con.commit()
    con.close()

init_db()

# Helper: Add folder to DB
def add_folder_to_db(folder: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO folders (name) VALUES (?)", (folder,))
    con.commit()
    con.close()

# Helper: Get folders from DB
def load_folders_from_db() -> list:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT name FROM folders")
    folders = [row[0] for row in cur.fetchall()]
    con.close()
    return folders

# Helper: Add file record into DB
def add_file_to_db(folder: str, filename: str, file_path: str, extracted_text: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO files (folder, filename, file_path, extracted_text)
        VALUES (?, ?, ?, ?)
    """, (folder, filename, file_path, extracted_text))
    con.commit()
    con.close()

# Helper: Load files for a folder from DB
def load_files_for_folder(folder: str) -> Dict[str, dict]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT filename, file_path, extracted_text FROM files WHERE folder = ?", (folder,))
    files = {row[0]: {"path": row[1], "text": row[2]} for row in cur.fetchall()}
    con.close()
    return files

# Helper: Embedding via Azure OpenAI
def get_embeddings(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [e.embedding for e in resp.data]

# Persistence: per-folder FAISS index and metadata
class FolderIndex:
    def __init__(self, folder: str, dim: int = 1536):
        self.folder = folder
        self.index_path = os.path.join(PERSIST_DIR, f"{folder}_index.faiss")
        self.meta_path = os.path.join(PERSIST_DIR, f"{folder}_meta.pkl")
        # load or init
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f: self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []  # list of dicts per vector
    def add(self, embeddings: List[List[float]], metas: List[dict]):
        arr = np.array(embeddings, dtype="float32")
        self.metadata.extend(metas)
        self._save()
    def query(self, embedding: List[float], k: int = 5):
        vec = np.array([embedding], dtype="float32")
        D, I = self.index.search(vec, k, None, None,None)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.metadata):
                results.append({**self.metadata[idx], "score": float(dist)})
        return results
    def delete(self, filename: str):
        # soft-delete by marking metadata, not removing from index
        for m in self.metadata:
            if m.get("source") == filename:
                m["deleted"] = True
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

# ─── 3. Helper functions ─────────────────────────────────────────────────────
def extract_with_azure(file_bytes: bytes) -> str:
    poller = doc_client.begin_analyze_document("prebuilt-document", file_bytes)
    result = poller.result()
    return "\n\n".join([p.content for p in result.paragraphs or []])

def extract_local(doc_bytes: bytes, mime: str) -> str:
    if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(BytesIO(doc_bytes))
    elif mime in ("application/pdf", "image/png", "image/jpeg"):
        return pytesseract.image_to_string(Image.open(BytesIO(doc_bytes)))
    elif mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(BytesIO(doc_bytes), engine="openpyxl")
        return df.to_csv(index=False)
    return ""

def analyze_with_gpt(messages: list) -> str:
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.2,
        max_tokens=1000
    )
    return response.choices[0].message.content or ""

# Generate PDF for last assistant reply
def generate_pdf(content: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 40, height - 80, width=100, preserveAspectRatio=True)
    text_obj = c.beginText(40, height - 100)
    for line in content.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.showPage()
    c.save()
    return buffer.getvalue()

# Extract up to 4 key snapshots from raw text
def ai_snapshots(text: str) -> List[Dict[str,str]]:
    """
    Asks the model to return exactly three items, each with:
      - a “title” (topic name)
      - a one-line “excerpt” pulled verbatim from the document.
    Returns a list of dicts: [{ "title": "...", "excerpt": "..." }, …]
    """
    system = "You are an assistant that extracts the three most important topics from the document and for each gives one representative sentence verbatim."
    user_prompt = (
        "Document:\n"
        f"{text[:3000]}\n\n"                      # limit to first 3k chars
        "Please output exactly 3 items in JSON array form, "
        "each with keys `title` and `excerpt`.  "
        "Example:\n"
        '[{"title":"Revenue Forecast","excerpt":"In Q2 we expect growth of 15%..."}]'
    )
    raw = analyze_with_gpt([{"role":"system","content":system},
                            {"role":"user","content":user_prompt}])
    try:
        return json.loads(raw)
    except:
        # fallback: parse lines loosely
        items = []
        for block in raw.split("\n\n")[:3]:
            m = re.match(r"^\d+[\.\)]\s*(.+?):\s*(“?)(.+?)(“?)$", block.strip())
            if m:
                items.append({"title":m.group(1), "excerpt":m.group(3)})
        return items


# ─── 4. Streamlit UI ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Chabot", layout="wide")

# Initialize session state variables early.
if "folders" not in st.session_state:
    st.session_state.folders = load_folders_from_db()  # or [] if no folders exist
    st.session_state.current_folder = None
    st.session_state.indexes = {}
    st.session_state.docs = {}
    st.session_state.selected_files = {}
    st.session_state.chat_histories = {}

# Sidebar header
st.sidebar.markdown("<h1 style='color:#6200ce;'>ChatBot</h1>", unsafe_allow_html=True)

# Sidebar: projects
st.sidebar.header("Projects")
if st.session_state.folders:
    selected_project = st.sidebar.selectbox(
        "Select Project",
        options=st.session_state.folders,
        index=st.session_state.folders.index(st.session_state.current_folder)
            if st.session_state.current_folder in st.session_state.folders else 0
    )
    st.session_state.current_folder = selected_project

# Sidebar: folders & files
st.sidebar.header("📂 Folders & Files")
if "folders" not in st.session_state:
    st.session_state.folders = load_folders_from_db()  # or [] if you prefer
    st.session_state.current_folder = None
    st.session_state.indexes = {}
    st.session_state.docs = {}
    st.session_state.selected_files = {}
    st.session_state.chat_histories = {}

# Load existing on startup
for fname in os.listdir(PERSIST_DIR):
    if fname.endswith("_index.faiss"):
        folder = fname.replace("_index.faiss", "")
        st.session_state.folders.append(folder)
        st.session_state.indexes[folder] = FolderIndex(folder)
        # load chat
        chat_path = os.path.join(PERSIST_DIR, f"{folder}_chat.pkl")
        if os.path.exists(chat_path):
            with open(chat_path, "rb") as f:
                st.session_state.chat_histories[folder] = pickle.load(f)
        else:
            st.session_state.chat_histories[folder] = []

# Sidebar: Folder navigation
st.sidebar.header("Conversations & Folders")
new_folder = st.sidebar.text_input("New folder name")
if st.sidebar.button("Add Folder") and new_folder:
    if new_folder not in st.session_state.folders:
        st.session_state.folders.append(new_folder)
        add_folder_to_db(new_folder)
        st.session_state.docs[new_folder] = {}
        st.session_state.selected_files[new_folder] = set()
    st.session_state.current_folder = new_folder

for folder in st.session_state.folders:
    if st.sidebar.button(folder):
        st.session_state.current_folder = folder
    # Initialize docs and select state
    st.session_state.docs.setdefault(folder, {})
    st.session_state.selected_files.setdefault(folder, set())

    # Load files text from DB for the folder if not yet loaded
    if not st.session_state.docs[folder]:
        st.session_state.docs[folder] = {
            fname: fileinfo["text"] for fname, fileinfo in load_files_for_folder(folder).items()
        }
        # You may also want to mark selected files based on your UI logic:
        st.session_state.selected_files[folder] = set(load_files_for_folder(folder).keys())

# Display folder-specific file uploader and file list in the sidebar
for folder in st.session_state.folders:
    with st.sidebar.expander(folder, expanded=(folder == st.session_state.current_folder)):
        if st.button(f"Select '{folder}'", key=f"sel_{folder}"):
            st.session_state.current_folder = folder
        st.write("**Files:**")
        files = list(st.session_state.docs.get(folder, {}).keys())
        for fname in files:
            checked = fname in st.session_state.selected_files.get(folder, set())
            new = st.checkbox(fname, value=checked, key=f"chk_{folder}_{fname}")
            if new:
                st.session_state.selected_files[folder].add(fname)
            else:
                st.session_state.selected_files[folder].discard(fname)
        uploaded = st.file_uploader(
            f"Upload to '{folder}' (PDF/DOCX/XLSX/PNG/JPG)",
            type=["pdf","docx","xlsx","png","jpg"],
            accept_multiple_files=True,
            key=f"up_{folder}"
        )
        if uploaded:
            for f in uploaded:
                # Only add if this file isn't already stored and within size limit.
                if f.name not in st.session_state.docs[folder] and f.size <= MAX_SIZE_MB*1024*1024:
                    raw = f.read()
                    # Prepare folder-specific upload directory
                    folder_upload_dir = os.path.join(UPLOAD_DIR, folder)
                    os.makedirs(folder_upload_dir, exist_ok=True)
                    file_path = os.path.join(folder_upload_dir, f.name)
                    # Save the actual file
                    with open(file_path, "wb") as out_file:
                        out_file.write(raw)
                    mime = f.type
                    try:
                        text = extract_with_azure(raw)
                        if not text.strip():
                            raise ValueError
                    except Exception:
                        text = extract_local(raw, mime)
                    # Update session_state docs and selected files
                    st.session_state.docs[folder][f.name] = text
                    st.session_state.selected_files[folder].add(f.name)
                    # Persist file record in database
                    add_file_to_db(folder, f.name, file_path, text)

# Main layout: two columns with a styled right-hand drawer
st.markdown(
    """
    <style>
    .vertical-divider {
                       /* above the page content */
    }
    </style>
    <div class="vertical-divider"></div>
    """,
    unsafe_allow_html=True
)
chat_col, right_col = st.columns([3,1])
with chat_col:
    
    cf = st.session_state.current_folder

    if cf is None:
        # Show welcome banner
        st.image("banner.png",  use_container_width=True)
        st.image("banner2.png", use_container_width=True)
    else:
        # Render the conversation history
        for msg in st.session_state.chat_histories.get(cf, []):
            st.chat_message(msg["role"]).write(msg["content"])
        
        # Place the chat input bar at the bottom
        new_input = st.chat_input("Ask a question…")
        if new_input:
            # Append the user message to the conversation history
            st.session_state.chat_histories.setdefault(cf, []).append({"role": "user", "content": new_input})
            
            texts = [st.session_state.docs[cf][fn] for fn in st.session_state.selected_files.get(cf, set())]
            context = "".join(texts)
            msgs = [{"role": "system", "content": "You are a helpful document analysis assistant."}]
            if context:
                msgs.append({"role": "system", "content": context})
            msgs.append({"role": "user", "content": new_input})
            
            with st.spinner("Analyzing…"):
                reply = analyze_with_gpt(msgs)
                
            # Append the assistant reply and force a rerun to update the chat display
            st.session_state.chat_histories[cf].append({"role": "assistant", "content": reply})
            st.rerun()

with right_col:
    tabs = st.tabs(["Selected Documents","Source Snapshots"])
    with tabs[0]:
         st.markdown(
    """
    <style>
      /* container to stack cards */
      .doc-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 0;
        margin: 0;
      }
      /* each card */
      .doc-card {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-size: 0.875rem;
      }
      /* icon on the left */
      .doc-card .icon {
        margin-right: 8px;
        font-size: 1.1rem;
        color: #6200ee;
      }
      /* filename will ellipsize if too long */
      .doc-card .filename {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
    </style>
    """,
    unsafe_allow_html=True
    )
    if cf and st.session_state.selected_files.get(cf):
        files = list(st.session_state.selected_files[cf])
        # Build one HTML block without extra newlines or spaces.
        cards_html = "".join(
            f'<div class="doc-card"><span class="icon">📄</span><span class="filename">{fname}</span></div>'
            for fname in files
        )
        st.markdown(f"<div class='doc-container'>{cards_html}</div>", unsafe_allow_html=True)
    else:
        st.write("No files selected.")
    with tabs[1]:
       st.markdown("### Source Snapshots")
       if cf:
        text = "".join(st.session_state.docs.get(cf, {}).values())
        snaps = ai_snapshots(text)   # ← new!
        if snaps:
            for i,item in enumerate(snaps, start=1):
                st.markdown(f"**{i}. {item['title']}**")
                st.write(item['excerpt'])
                with st.expander(f"Snapshot: {item['title']}"):
                    st.write(
                        "Excerpt from your document—this is the page/context where "
                        "the heading was pulled:\n\n"
                        + item['excerpt']
                    )
        else:
            st.write("No snapshots could be generated.")
       else:
         st.write("Select a folder to view snapshots.")

