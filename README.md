# RAG Agent PDF Ingestion

A simple Retrieval-Augmented Generation (RAG) pipeline built with LangChain and OpenAI that ingests PDF documents, splits them into chunks, creates embeddings, stores them in a vector database, and answers questions using retrieved context.

## Features

- Load and parse PDF documents
- Extract text page by page
- Split documents into semantic chunks
- Generate embeddings using OpenAI
- Store embeddings in a vector database
- Perform similarity search
- Answer questions using Retrieval-Augmented Generation (RAG)

---

## Project Structure

```text
rag-agent-pdf-ingestion/
│
├── data/
│   ├── consumer-privacy-policy.pdf
│   ├── online-privacy-policy.pdf
│   └── ...
│
├── pdf_loader.py
├── document_splitter.py
├── embeddings.py
├── main.py
│
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

---

## Pipeline Overview

### 1. PDF Loading

The `PDFLoader` class:

- Reads all PDF files from the `data/` directory
- Extracts text page by page
- Creates LangChain `Document` objects
- Attaches metadata such as:
  - source file
  - page number

### 2. Document Splitting

The `DocumentSplitter` class:

- Uses `RecursiveCharacterTextSplitter`
- Splits large pages into smaller chunks
- Maintains document metadata

Default configuration:

```python
chunk_size = 1000
chunk_overlap = 150
```

### 3. Embedding Generation

The `Embeddings` class:

- Converts document chunks into vector embeddings
- Creates a vector store for retrieval

### 4. Retrieval

For each user query:

```python
results = vectorstore.similarity_search(
    question,
    k=3
)
```

The top matching chunks are retrieved and used as context.

### 5. Question Answering

The retrieved context is sent to an OpenAI model along with the user's question.

The model generates answers grounded in the source documents.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/VinayKamath/rag-agent-pdf-ingestion.git

cd rag-agent-pdf-ingestion
```

### Create a Virtual Environment

Using `uv`:

```bash
uv venv
```

Activate:

#### Windows

```bash
.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

### Install Dependencies

Using `uv`:

```bash
uv sync
```

Or:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

Do not commit this file to GitHub.

---

## Running the Project

Place PDF files inside the `data/` directory.

Run:

```bash
python main.py
```

Example query:

```python
question = "Can you summarize the CHASE consumer policy?"
```

Example output:

```text
The CHASE Consumer Privacy Policy explains how customer
information is collected, used, shared, and protected...
```

---

## Example Flow

```text
PDF Documents
      │
      ▼
 PDF Loader
      │
      ▼
 Document Splitter
      │
      ▼
 OpenAI Embeddings
      │
      ▼
 Vector Store
      │
      ▼
 Similarity Search
      │
      ▼
 GPT Model
      │
      ▼
 Generated Answer
```

---

## Technologies Used

- Python
- LangChain
- OpenAI
- PyMuPDF
- ChromaDB
- UV
- Jupyter Notebook

---

## Future Improvements

- Conversational memory
- LangGraph integration
- Streaming responses
- Metadata filtering
- Hybrid search
- Multi-document QA
- Source citations in responses
- Web interface with Streamlit

---

## Author

**Vinay Kamath**

GitHub: https://github.com/VinayKamath

---

## License

This project is intended for educational and experimentation purposes.
