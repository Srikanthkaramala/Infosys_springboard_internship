# Policy RAG Assistant (AI Module)

> **Submodule:** `ai/` — AI-powered Policy Question Answering System using LM Studio, LangChain, and FAISS  

This module implements a **Retrieval-Augmented Generation (RAG)**–based intelligent assistant that answers questions about government or institutional policies.  
It integrates **local LLM inference (TinyLlama via LM Studio)**, **FAISS vector search**, and a **Streamlit-based interface** for interactive querying.

---

## Key Features

-  **RAG Pipeline** — Context-aware responses grounded in your policy dataset  
-  **Local Model Inference** — Runs fully offline through **LM Studio**  
-  **Voice + Text Input** — Interact naturally by typing or speaking  
-  **Multilingual Output** — Translate responses into multiple languages  
-  **Text-to-Speech (TTS)** — Listen to answers with audio playback  
-  **Download Options** — Export results as PDF or JSON

---

## Folder Structure

```

project_root/
│
├── ai/
│   ├── app.py                 # Streamlit frontend
│   ├── build_vectorstore.py   # Backend script for FAISS index creation
│   ├── data/
│   │   ├── policies.csv       # Input dataset
│   │   └── policy_index/      # Generated FAISS embeddings
│   │       ├── index.faiss
│   │       └── index.pkl
│   └── README.md             
````

---

## Setup Guide

# Create and Activate Virtual Environment

```bash
cd ai
python -m venv venv
venv\Scripts\activate      # Windows
# OR
source venv/bin/activate   # macOS/Linux
````

# Install Dependencies

```bash
pip install -U streamlit fpdf gTTS SpeechRecognition deep-translator \
langchain langchain-core langchain-openai langchain-community \
langchain-text-splitters sentence-transformers
```

# Configure LM Studio

1. Open **[LM Studio](https://lmstudio.ai/)**
2. Load model: `tinyllama-1.1b-chat-v1.0` (or another chat-compatible model)
3. Start the **local API server** → default:

   ```
   http://localhost:1234/v1
   ```
4. Keep LM Studio running in the background.

---

#  Step 1: Build FAISS Vector Index

Run once to embed and store your policy data:

```bash
python build_vectorstore.py
```

This script:

* Loads `data/policies.csv`
* Splits long text into smaller chunks
* Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
* Saves FAISS index locally at `data/policy_index/`

---

# Step 2: Launch Streamlit App

```bash
streamlit run app.py
```

Then open the provided URL in your browser.
You can:

* Type or speak your query
* Choose an output language
* Listen to or download the generated answer

---

# CSV Input Format

| policy_id | title                | full_text                       | state | year | category  | status | region |
| --------- | -------------------- | ------------------------------- | ----- | ---- | --------- | ------ | ------ |
| 1         | Education Reform Act | Full text of the policy here... | Delhi | 2021 | Education | Active | North  |

Ensure that your dataset is stored in `data/policies.csv`.

---

## 🧰 Tech Stack

| Component                     | Description                    |
| ----------------------------- | ------------------------------ |
| **LangChain**                 | Orchestrates RAG pipeline      |
| **FAISS**                     | Vector-based similarity search |
| **TinyLlama (via LM Studio)** | Local LLM inference            |
| **Streamlit**                 | Interactive web interface      |
| **Sentence Transformers**     | Embedding generation           |
| **gTTS / Deep Translator**    | Audio and multilingual support |
| **SpeechRecognition**         | Voice query functionality      |

---
# Example Queries

* “What are the key education policies in India?”
* “List healthcare schemes active in Tamil Nadu.”
* “Explain renewable energy initiatives.”
---

## 🧩 Environment Configuration

Set automatically within the app:

```bash
OPENAI_API_BASE=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
```

## 🧭 Credits

* [LM Studio](https://lmstudio.ai/)
* [LangChain](https://www.langchain.com/)
* [Sentence Transformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Streamlit](https://streamlit.io/)


