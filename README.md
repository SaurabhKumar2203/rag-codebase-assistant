# 🤖 Multi-Repo AI Codebase Assistant

An advanced, production-grade Retrieval-Augmented Generation (RAG) application that allows you to chat with any public Python GitHub repository. Built with **Streamlit**, **LlamaIndex**, **Pinecone**, **HuggingFace**, and **Google Gemini**.

---

## 📸 Application Preview

> *Here is a preview of the Multi-Repo AI Codebase Assistant interface:*

<img width="1915" height="913" alt="image" src="https://github.com/user-attachments/assets/e3053da0-bf95-4e71-b383-d0ec0fb9b95e" />



---

## 🚀 Key Features

* **AST-Aware Code Chunking:** Uses Python Abstract Syntax Tree (AST) code splitting (`CodeSplitter`) to preserve functional boundaries, classes, and methods instead of arbitrary text truncation.
* **Local Embeddings:** Utilizes HuggingFace (`bge-small-en-v1.5`) locally on your CPU for high-performance, cost-free vector generation.
* **High-Precision Cross-Encoder Reranker:** Integrates a cross-encoder model (`ms-marco-MiniLM-L-2-v2`) to deeply re-score candidate chunks contextually, guaranteeing only the sharpest code context is handed to the LLM.
* **Stateful Conversational Memory:** Powered by Google Gemini and LlamaIndex's `condense_plus_context` engine, allowing natural multi-turn conversations and follow-up questions.
* **Isolated Multi-Repo Management:** Leverages Pinecone serverless namespaces to cleanly separate and switch between different repositories on the fly.
* **Transparent Citations:** Expandable source file cards detailing exact file paths and rerank relevance scores for every AI response.

---

## 🛠️ Tech Stack

* **UI Framework:** Streamlit
* **Orchestration & RAG:** LlamaIndex
* **Vector Database:** Pinecone (Serverless)
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
* **Reranker:** Cross-Encoder (`ms-marco-MiniLM-L-2-v2`)
* **LLM:** Google Gemini (`gemini-3.5-flash`)

---

## 📁 Project Structure

```text
rag-codebase-assistant/
│
├── app.py                 # Main Streamlit web application
├── requirements.txt       # Project dependencies
├── test_generation.py     # Local CLI RAG testing script
└── README.md              # Project documentation

```

---

## ⚙️ Local Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SaurabhKumar2203/rag-codebase-assistant.git
cd rag-codebase-assistant

```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure Environment Variables

Set your API keys as environment variables in your terminal:

```bash
# On Windows (PowerShell):
$env:PINECONE_API_KEY="your_pinecone_api_key"
$env:GOOGLE_API_KEY="your_google_api_key"

# On macOS/Linux:
export PINECONE_API_KEY="your_pinecone_api_key"
export GOOGLE_API_KEY="your_google_api_key"

```

### 5. Run the Streamlit Web App

```bash
streamlit run app.py

```

---

## ☁️ Cloud Deployment (Streamlit Community Cloud)

1. Push your project to a public or private GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and click **Create app**.
3. Link your repository, set the branch to `main`, and main file path to `app.py`.
4. Open **Advanced settings -> Secrets** and add your credentials in TOML format:
```toml
PINECONE_API_KEY = "your_pinecone_api_key"
GOOGLE_API_KEY = "your_google_api_key"

```


5. Click **Deploy!**
