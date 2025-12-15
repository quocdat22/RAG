```mermaid
graph TB
    subgraph "Data Ingestion Layer"
        A[Documents Sources] --> B[Document Loaders]
        B --> C{Document Type}
        C -->|PDF| D[PDFReader]
        C -->|CSV/Excel| E[PandasReader]
        C -->|JSON| F[JSONReader]
        C -->|SQL| G[SQLReader]
        D --> H[Text Splitter]
        E --> H
        F --> H
        G --> H
        H --> I[Chunk + Metadata]
    end

    subgraph "Embedding & Storage Layer"
        I --> J[Embedding Model]
        J -->|OpenAI/HuggingFace| K[Vector Embeddings]
        K --> L[(Chroma VectorDB)]
        I --> M[(Document Store)]
        M -.metadata.-> L
    end

    subgraph "Query Processing Layer"
        N[User Query] --> O[Query Transformer]
        O --> P{Query Type}
        P -->|Simple| Q[Basic Retriever]
        P -->|Complex| R[Advanced Retriever]
        P -->|Analytical| S[SQL/Pandas Query]
        
        Q --> T[Similarity Search]
        R --> U[Hybrid Search]
        R --> V[Re-ranking]
        V --> W[Top-K Results]
        U --> W
        T --> W
        
        S --> X[Structured Data Query]
    end

    subgraph "LlamaIndex Core"
        L --> T
        L --> U
        M --> X
        
        W --> Y[Context Builder]
        X --> Y
        Y --> Z[Prompt Template]
        Z --> AA[LLM OpenAI/Anthropic]
        AA --> AB[Response Synthesizer]
    end

    subgraph "Analysis & Output Layer"
        AB --> AC{Output Type}
        AC -->|Text Answer| AD[Formatted Response]
        AC -->|Chart/Graph| AE[Data Visualization]
        AC -->|Table| AF[Structured Output]
        AC -->|Report| AG[Multi-step Analysis]
        
        AG --> AH[Chain of Thought]
        AH --> AI[Intermediate Results]
        AI --> AJ[Final Report]
    end

    subgraph "Monitoring & Optimization"
        AA -.usage.-> AK[Token Counter]
        AB -.quality.-> AL[Response Evaluator]
        T -.relevance.-> AM[Retrieval Metrics]
        
        AK --> AN[(Logs & Analytics)]
        AL --> AN
        AM --> AN
    end

    subgraph "Caching & Performance"
        N -.check.-> AO{Cache Hit?}
        AO -->|Yes| AP[Return Cached]
        AO -->|No| O
        AB --> AQ[Update Cache]
    end

    AD --> AR[User Interface]
    AE --> AR
    AF --> AR
    AJ --> AR
    AP --> AR

    style L fill:#e1f5ff
    style AA fill:#fff4e1
    style AN fill:#f0f0f0
    style AR fill:#e8f5e9