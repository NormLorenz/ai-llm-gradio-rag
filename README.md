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
    ┌──────┴───────┐
    │              │
    ▼              ▼
┌─────────┐  ┌──────────────┐
│ Question│  │ RetrievalQA  │
│ Input   │  │ Chain        │
└─────────┘  │ (GPT-4)      │
    │        └──────────────┘
    │              │
    └──────┬───────┘
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

### Steps:
1. Upload a PDF or TXT document
2. (Optional) Adjust chunk size and overlap in Advanced Settings
3. Click "Process Document"
4. Ask questions about the document content
5. View answers with source citations

## Configuration

### Chunk Size and Overlap
- **Chunk Size**: Number of characters per text segment (default: 1000, range: 100-2000)
- **Chunk Overlap**: Character overlap between consecutive chunks for context preservation (default: 200, range: 0-500)

### LLM Settings
- **Model**: GPT-4 with temperature 0 for deterministic responses
- **Retrieval**: Top-3 most relevant document chunks
- **Vector Dimension**: 1536 (OpenAI embedding dimension)

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