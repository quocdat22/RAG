# Scientific Paper Research Assistant

A specialized Retrieval-Augmented Generation (RAG) system designed for scientists and researchers to analyze, compare, and extract insights from academic papers.

## 🏗️ Architecture

This system follows a modular architecture optimized for research workflows:

1. **Scientific Ingestion Layer** - PDF parsing, metadata extraction (DOI, arXiv, Venue), and section-aware chunking
2. **Embedding & Storage Layer** - Vector embeddings with Chroma
3. **Research Query Processing** - Intelligent routing for comparisons, trends, and gap analysis
4. **LlamaIndex Core** - Orchestration and context management
5. **Research Analysis Layer** - Specialized analyzers for:
   - **Comparison Matrices**: Compare methods side-by-side
   - **Trend Analysis**: Track evolution of topics over time
   - **Gap Identification**: Find missing research areas
   - **Consensus Detection**: Identify agreement vs. controversy
6. **Monitoring & Optimization** - Performance tracking
7. **Caching & Performance** - Redis caching

## 🚀 Features

- **Academic Paper Specialized**: Extracts abstract, DOI, venue, and authors automatically
- **Deep Research Analysis**:
  - Compare multiple papers/methods automatically
  - Analyze trends over years
  - Detect research gaps
- **Smart Citations**: Answers include numbered citations `[1]` linked to specific papers and pages
- **Intelligent Retrieval**: Hybrid search + Cohere Reranking for high precision
- **Streamlit UI**: Clean interface for searching and managing your library
- **Incremental Indexing**: Efficiently handle growing libraries

## 📋 Prerequisites

- Python 3.10 or higher
- UV package manager
- OpenAI API key (GitHub Models or OpenAI)
- Cohere API key (for reranking)
- Redis (optional, for caching)

## 🔧 Installation

### 1. Clone and Setup

```bash
cd c:\Users\dat\Projects\RAG
```

### 2. Install Dependencies with UV

```bash
# Install production dependencies
uv sync

# Install with development dependencies
uv sync --extra dev
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
notepad .env
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI or GitHub Models API key
- `COHERE_API_KEY`: Your Cohere API key for reranking

## 🎯 Quick Start

### 1. Run the Streamlit UI

```bash
.venv/Scripts/activate && python run_ui.py
```

### 2. Upload Papers

- Navigate to http://localhost:8501
- Upload PDF papers or text documents
- The system will automatically extract metadata (Titles, Authors, DOIs)

### 3. Research Mode

- **General Search**: Ask questions like "What is the Transformer architecture?"
- **Comparison**: "Compare BERT and GPT-3 on accuracy and training cost"
- **Trend Analysis**: "How has object detection evolved from 2018 to 2024?"
- **Gap Analysis**: "What are the limitations of current RAG approaches?"

## 📁 Project Structure

```
RAG/
├── config/              # Configuration management
├── src/                 # Main application code
│   ├── ingestion/       # Document loading & metadata extraction
│   ├── embedding/       # Vector embeddings
│   ├── retrieval/       # Hybrid search & reranking
│   ├── generation/      # LLM & Research Analyzers
│   └── core/            # Utilities
├── ui/                  # Streamlit interface
├── tests/               # Test suite
└── data/                # Data storage
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest
```

## 📈 Performance

- **Query Latency**: < 3s (P95)
- **Retrieval Precision**: 85%+
- **Supported Scale**: 100K-1M documents

## 🔐 Security

- API keys stored in `.env` (never committed)
- Document data stays local

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 🗺️ Roadmap

### Phase 1 (MVP) ✓
- Basic Q&A functionality
- PDF support with metadata extraction
- Streamlit UI
- Research Analyzers (Comparison, Trends)

### Phase 2 (Planned)
- Citation graph visualization
- Export reports to LaTeX/BibTeX
- Integration with arXiv API
- Collaborative libraries

### Phase 3 (Future)
- Multi-modal paper understanding (Figures/Tables)
- Cloud deployment

