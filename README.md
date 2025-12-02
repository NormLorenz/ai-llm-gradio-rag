# 📚 RAG Q&A Pipeline

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions about their content. The application uses advanced language models and vector databases to provide accurate, source-backed answers.

## Overview

This application combines state-of-the-art technologies to create an intelligent document question-answering system. Users can upload PDF or TXT files, and the system processes them into semantic chunks that are indexed in a vector database. When users ask questions, the system retrieves relevant document segments and uses a large language model to generate accurate answers with source citations.

## Key Features

- 📄 **Multi-format Document Support**: Upload PDF or TXT files
- 💬 **AI-Powered Q&A**: Ask natural language questions about document content
- 🔗 **Source Attribution**: Get answers with citations from the original document
- ⚙️ **Advanced Settings**: Customize chunk size and overlap for optimal processing
- 🚀 **Real-time Processing**: Instant document indexing and query responses
- 📚 **Load Existing Data**: Skip reprocessing by loading previously indexed documents from Pinecone

## Technologies Used

### Core Framework
- **[Gradio](https://www.gradio.app/)** - Modern web interface for machine learning models and data applications

### AI & Language Processing
- **[OpenAI](https://openai.com/)** - GPT-4 language model for intelligent question answering
- **[LangChain](https://www.langchain.com/)** - Framework for building applications with large language models
  - `langchain-core` - Core abstractions and interfaces
  - `langchain-classic` - Classic chains for retrieval-based QA
  - `langchain-community` - Community integrations for document loaders
  - `langchain-openai` - OpenAI integration (embeddings and chat models)
  - `langchain-text-splitters` - Recursive text splitting for optimal chunk management
  - `langchain-pinecone` - Pinecone vector store integration

### Vector Database & Embeddings
- **[Pinecone](https://www.pinecone.io/)** - Serverless vector database for semantic search
- **[OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)** - Text-to-vector conversion for semantic similarity

### Document Processing
- **[PyPDF](https://pypdf.readthedocs.io/)** - PDF document extraction and parsing
- **Python Built-in** - Text file loading

### Utilities
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** - Environment variable management for API keys

## Architecture

```
┌─────────────────────┐
│  Document Upload    │
│  (PDF/TXT)          │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│  Text Chunking       │
│  (RecursiveCharText  │
│   TextSplitter)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Embedding           │
│  (OpenAI)            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Vector Store        │
│  (Pinecone)          │
└──────────┬───────────┘
           │
    ┌──────┴───────────┐
    │                  │
    ▼                  ▼
┌─────────────┐  ┌──────────────┐
│ Load Query  │  │ New Query    │
│ Existing    │  │ RetrievalQA  │
│ VectorStore │  │ Chain        │
└─────────────┘  │ (GPT-4)      │
    │            └──────────────┘
    │                  │
    │            ┌─────┴
    │            │
    └────────┬───┘
             ▼
    ┌────────────────┐
    │ Answer with    │
    │ Sources        │
    └────────────────┘
```

## How It Works

1. **Document Processing**: When a user uploads a document, the system extracts text and splits it into semantic chunks
2. **Embedding Creation**: Each chunk is converted into a vector representation using OpenAI embeddings
3. **Vector Indexing**: Vectors are stored in Pinecone for efficient similarity search
4. **Query Processing**: When a question is asked, it's converted to a vector and similar document chunks are retrieved
5. **Answer Generation**: The retrieved chunks and question are sent to GPT-4, which generates a contextual answer
6. **Source Attribution**: The system includes relevant document excerpts to back up the answer

## Installation

### Prerequisites
- Python 3.9+
- OpenAI API Key
- Pinecone API Key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-llm-gradio-rag
```

2. Install dependencies using `uv`:
```bash
uv sync
```

3. Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

## Usage

Run the application:
```bash
uv run main.py
```

The Gradio interface will open in your browser (default: `http://localhost:7860`).

### Processing New Documents

1. Upload a PDF or TXT document using the file upload box
2. (Optional) Adjust chunk size and overlap in Advanced Settings for fine-tuned processing
3. Click **"🚀 Process Document"** to process and index the document
4. Wait for the status message confirming successful processing
5. Ask questions about the document content

### Loading Previously Indexed Documents

If you've already processed and indexed documents in Pinecone, you can skip the processing step:

1. Click **"📚 Load Existing Data"** to connect to Pinecone and load previously indexed vectors
2. The system will display the number of vectors in the index
3. Proceed directly to asking questions without reprocessing

### Asking Questions

1. Enter your question in the "Your Question" text box
2. Click **"🔍 Get Answer"** or press Enter
3. View the AI-generated answer with relevant source citations from the document

## Configuration

### Chunk Size and Overlap
- **Chunk Size**: Number of characters per text segment (default: 1000, range: 100-2000)
- **Chunk Overlap**: Character overlap between consecutive chunks for context preservation (default: 200, range: 0-500)

### LLM Settings
- **Model**: GPT-4 with temperature 0 for deterministic responses
- **Retrieval**: Top-3 most relevant document chunks
- **Vector Dimension**: 1536 (OpenAI embedding dimension)

## Core Methods

### `process_document(file, chunk_size, chunk_overlap)`
Processes a new document and indexes it in Pinecone:
- Loads PDF or TXT files
- Splits text into semantic chunks
- Creates embeddings using OpenAI
- Stores vectors in Pinecone vector database
- Initializes the QA chain for answering questions

### `load_existing_vectorstore()`
Loads a previously indexed vector store from Pinecone:
- Connects to Pinecone
- Checks if the index contains existing vectors
- Loads the vector store without reprocessing documents
- Initializes the QA chain for immediate questioning
- Returns the number of vectors in the index

**Use Case**: When you've already indexed documents in Pinecone and want to use them without reprocessing.

### `answer_question(question)`
Answers a question using the loaded vector store:
- Retrieves the top-3 most relevant document chunks
- Sends chunks and question to GPT-4
- Returns answer with source citations
- Handles edge cases (no document loaded, empty questions)

## Project Structure

```
ai-llm-gradio-rag/
├── main.py                 # Main application file
├── main.ipynb             # Jupyter notebook version
├── pyproject.toml         # Project configuration and dependencies
├── README.md              # This file
└── LICENSE
```

## Dependencies

See `pyproject.toml` for the complete list of dependencies.