Chatbot Tech Stack & Quick Build Guide
A compact, production-ready chatbot stack that avoids training a custom LLM.
This repo uses hosted LLM APIs (OpenAI / Google Gemini), Azure Document Intelligence for PDF/form parsing, and a Streamlit front-end for demos and internal usage.

Dev environment (local)
IDE: VS Code (recommended extensions: Python, Pylance, Docker, Live Share)

Python: >= 3.10 — backend logic, API clients, orchestration. Use venv or Poetry for dependency isolation.

Suggested local setup

bash
Copy
Edit
# create & activate virtualenv (macOS / Linux)
python -m venv .venv
source .venv/bin/activate

# (Windows - CMD)
.venv\Scripts\activate

# (Windows - PowerShell)
.\.venv\Scripts\Activate.ps1

# install dependencies and open editor
pip install -r requirements.txt
code .
LLM / API layer
Approach: Use hosted LLM APIs — no custom model training required.

Options:

OpenAI (Chat Completions / Responses)

Google Gemini via MakerSuite (if available)

Why: faster iteration, lower infra cost, simpler to manage compliance when you limit what you send.

Notes: Verify pricing, quotas, and token handling for your chosen provider.

Example environment variables (.env)

ini
Copy
Edit
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
AZURE_FORM_RECOGNIZER_KEY=your_azure_key_here
AZURE_STORAGE_CONN_STRING=...
Document / Knowledge ingestion
Tool: Azure AI Document Intelligence (Form Recognizer) — extracts text, key-value pairs, tables from invoices, PDFs, forms, and scans.

Suggested workflow:

Upload raw files to blob storage (or pass file bytes directly).

Call Document Intelligence to extract text/fields.

Optional preprocessing: clean, chunk, and embed (if using retrieval/RAG).

Send cleaned context to the LLM for answer generation.

Tip: Evaluate free/trial limits and sanitize PII before sending data to third-party APIs.

Orchestration & API calling
Backend (Python): functions to:

call the document analyzer

build embeddings / retrieval (if RAG)

call LLM endpoints (OpenAI / Gemini)

orchestrate agentic workflows (optional)

HTTP layer (optional): FastAPI for a lightweight REST interface if you don’t want Streamlit to call the backend directly.

Small pseudo-example (Python)

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

print(resp.choices[0].message["content"])
Frontend / Demo UI
Streamlit: fast to spin up a chat UI for demos and stakeholder review.

Pattern: input box → async backend call (or direct) → display chat history + attachments.

When to replace: If you need production-grade UI, consider React or Vue with a hosted backend.

Deployment & infra
Cloud options: Azure App Service, Azure Container Instances, or AKS for scale.
For small demos: Streamlit Cloud or Vercel (static frontends + API elsewhere).

Secrets: store API keys in Azure Key Vault (or your cloud provider’s secrets manager).

CI/CD: GitHub Actions for linting, tests, build/push containers.

Security & compliance
Do not send sensitive/internal docs to external LLMs unless permitted.

Best practices:

remove or redact PII before API calls

consider private / on-prem endpoints if required

log minimal request/response metadata

enforce strict RBAC on storage and secrets

Quick checklist to reproduce
Install VS Code and create a Python virtual environment.

Create cloud accounts: Azure (Document Intelligence + storage) and your LLM provider (OpenAI / Google).

Implement backend pipeline: document ingestion → preprocess → (optional) embeddings/retrieval → call LLM.

Build Streamlit frontend to send prompts and display conversation.

Store secrets in env vars / Key Vault and deploy to Azure or your container platform.

Add CI/CD (GitHub Actions) to automate lint, test, and deploy.
