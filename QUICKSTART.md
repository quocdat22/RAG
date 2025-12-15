# Quick Start Guide

## Running the RAG System

### Option 1: Using run_ui.py (Recommended)
```bash
# Activate venv
.venv/Scripts/activate

# Set API key
$env:OPENAI_API_KEY="your-key"

# Run
python run_ui.py
```

### Option 2: Direct Streamlit
```bash
# From project root
.venv/Scripts/activate
$env:OPENAI_API_KEY="your-key"

# Add current directory to Python path
$env:PYTHONPATH = "."

# Run
streamlit run ui/app.py
```

### Option 3: Test Pipeline (No UI)
```bash
.venv/Scripts/activate
$env:OPENAI_API_KEY="your-key"

python examples/test_rag_pipeline.py
```

## First Time Setup

1. **Install dependencies** (already done ✅)
2. **Set API key**:
   ```bash
   $env:OPENAI_API_KEY="your-github-models-api-key"
   ```
3. **Run UI**:
   ```bash
   python run_ui.py
   ```
4. **Open browser**: http://localhost:8501

## Common Issues

### ModuleNotFoundError
- Make sure you're in project root directory
- Use `python run_ui.py` instead of direct streamlit

### API Key Not Found
```bash
# Set in PowerShell
$env:OPENAI_API_KEY="your-key"

# Or add to .env file
echo 'OPENAI_API_KEY=your-key' >> .env
```

### Port Already in Use
```bash
# Use different port
streamlit run ui/app.py --server.port 8502
```
