# SuperSearch

Chatbot Tech Stack & Quick Build Guide
Overview
I built a production-ready chatbot using a lightweight, practical stack that avoids training a custom LLM. The bot uses hosted LLM APIs, Azure document intelligence for parsing PDFs/forms, and a simple Streamlit front-end for demos and internal usage.

Dev environment (local)
VS Code — primary IDE (extensions: Python, Pylance, Docker, Live Share).

Python (>=3.10 recommended) — backend logic, API clients, and orchestration. Use a virtualenv or Poetry for dependency isolation.

Suggested local setup commands:

bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
code .
LLM / API layer
Use hosted LLM APIs — no need to train or host your own LLM. Options:

OpenAI APIs (Chat Completions / Responses)

Google Gemini via MakerSuite (if available for your account/region)

Why: faster to iterate, lower infrastructure cost, simpler compliance when you restrict the data you send.

Notes: check pricing, quotas, and token handling for whichever API you choose.

Environment variables (example):

ini
Copy
Edit
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
Document / Knowledge ingestion
Azure AI Document Intelligence (Form Recognizer) — extract structured data, key-value pairs, and text from invoices, PDFs, forms and scanned docs. Ideal for processing internal documents before feeding relevant content to the LLM.

Workflow:

Upload raw files to blob storage (or pass bytes).

Call Document Intelligence to extract text/fields.

Optionally run a text pre-processing step (clean, chunk, embed if using retrieval).

Send cleaned context to LLM API for answer generation.

Tip: evaluate free tier / trial limits and sanitize PII before sending to 3rd-party APIs.

Orchestration & API calling
Backend: Python functions to:

call the document analyzer,

handle embeddings or retrieval (if using RAG),

call LLM endpoints (OpenAI / Gemini),

orchestrate agent actions if you use agentic workflows.

HTTP layer: lightweight FastAPI (optional) if you want a REST interface rather than direct Streamlit calls.

Small pseudo-example for calling an LLM:

python
Copy
Edit
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
resp = openai.ChatCompletion.create(
  model="gpt-4o-mini",
  messages=[{"role":"user","content":"Summarize this doc..."}]
)
Frontend / Demo UI
Streamlit — quick to spin up a chat UI and demo the bot to stakeholders.

Use Streamlit for prototyping; if you need production UI, consider React/Vue + hosted backend.

You can generate a basic Streamlit chat UI quickly (search for “Streamlit chat demo” or use ChatGPT to scaffold code).

Minimal Streamlit pattern:

input box for user prompt

async call to backend or direct backend logic

display chat history and attachments

Deployment & infra
Cloud: Azure App Service, Azure Container Instances, or a container on AKS for scale. For small demos, Streamlit Cloud / Vercel (for static frontends + API elsewhere) also work.

Secrets: store API keys in Azure Key Vault or your cloud provider’s secrets manager.

CI/CD: GitHub Actions to run linting, tests, and build/push container images.

Security & compliance
Don't send sensitive/full internal docs to external LLMs unless allowed. Consider:

removing PII before API calls,

using on-prem or private endpoints where required,

logging minimal request/response metadata,

enforce strict RBAC on storage and keys.

Quick checklist to reproduce
Install VS Code and create a Python virtual environment.

Create cloud accounts (Azure for Document Intelligence + storage; LLM provider account for OpenAI / Google).

Implement backend: document ingestion → preprocess → call LLM.

Build Streamlit frontend to call the backend and display conversation.

Store keys in env vars / Key Vault; deploy to Azure or container platform.

