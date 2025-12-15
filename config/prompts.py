"""
Prompt Templates for RAG System

This module contains all LLM prompt templates used throughout the system.
Templates are organized by use case and follow best practices for prompt engineering.
"""

from typing import Any


# ============================================================================
# System Prompts
# ============================================================================

SYSTEM_PROMPT = """You are an expert data analyst AI assistant. Your role is to help users analyze documents and data to extract insights and answer questions accurately.

Guidelines:
1. Base your answers ONLY on the provided context - never make up information
2. Always cite sources using [doc_id] notation
3. If the information is not in the context, clearly state "I don't have enough information to answer this question."
4. Provide specific numbers, dates, and facts when available
5. Be concise but comprehensive
6. If asked for analysis, provide structured insights with evidence
7. Maintain a professional, helpful tone

Your responses should be:
- Accurate: Based strictly on provided context
- Clear: Easy to understand
- Cited: Always reference sources
- Helpful: Provide actionable insights when possible
"""

# ============================================================================
# Query Processing Prompts
# ============================================================================

QUERY_CLASSIFICATION_PROMPT = """Classify the following user query into one of these categories:

Categories:
1. SIMPLE - Straightforward factual questions that can be answered from single document
2. COMPLEX - Questions requiring analysis across multiple documents or reasoning
3. ANALYTICAL - Questions asking for trends, comparisons, or strategic insights

User Query: {query}

Respond with ONLY the category name (SIMPLE, COMPLEX, or ANALYTICAL).
"""

QUERY_EXPANSION_PROMPT = """Given the user's query, generate 2-3 alternative phrasings that might help retrieve relevant information.

Original Query: {query}

Provide alternative queries as a bullet list. Keep them concise and focused.
"""

# ============================================================================
# Response Generation Prompts
# ============================================================================

QA_RESPONSE_PROMPT = """Based on the context below, answer the user's question.

Context:
{context}

Metadata:
{metadata}

User Question:
{query}

Instructions:
1. Answer based ONLY on the provided context
2. Cite sources using [doc_id] after each claim
3. If the answer is not in the context, say "I don't have enough information about this."
4. Provide specific details (numbers, dates, names) when available
5. Keep the answer concise but complete

Answer:
"""

ANALYTICAL_RESPONSE_PROMPT = """Analyze the provided data and answer the user's analytical question.

Context:
{context}

Metadata:
{metadata}

User Question:
{query}

Instructions:
1. Provide a structured analysis with:
   - Key findings (bullet points)
   - Supporting evidence from context
   - Relevant trends or patterns
   - Data-driven insights
2. Cite all sources using [doc_id]
3. If data is insufficient, explicitly state limitations
4. Use clear sections and headings
5. Suggest visualizations if applicable (e.g., "This data would benefit from a line chart showing...")

Analysis:
"""

MULTI_STEP_ANALYSIS_PROMPT = """Conduct a comprehensive multi-step analysis to answer the user's question.

Context:
{context}

User Question:
{query}

Steps to follow:
1. Data Overview: Summarize the available data
2. Initial Analysis: Extract key statistics and facts
3. Deeper Insights: Identify patterns, correlations, anomalies
4. Recommendations: Provide actionable insights
5. Summary: Concise executive summary

For each step:
- Use clear headings
- Cite sources with [doc_id]
- Provide specific evidence
- Build logically on previous steps

Complete Analysis:
"""

# ============================================================================
# Document Summarization Prompts
# ============================================================================

DOCUMENT_SUMMARY_PROMPT = """Summarize the following document content.

Content:
{content}

Provide a concise summary (2-3 paragraphs) covering:
1. Main topic/purpose
2. Key points or findings
3. Important data or statistics
4. Conclusions or recommendations (if any)

Summary:
"""

METADATA_EXTRACTION_PROMPT = """Extract metadata from the following document content.

Content:
{content}

Extract and return as structured data:
1. Category: (e.g., financial_report, technical_doc, meeting_notes)
2. Key topics: (3-5 main topics)
3. Keywords: (5-10 important keywords)
4. Date references: (any dates mentioned)
5. Entities: (people, organizations, locations)

Metadata:
"""

# ============================================================================
# Error and Fallback Prompts
# ============================================================================

NO_CONTEXT_PROMPT = """I don't have any relevant documents to answer your question: "{query}"

This could be because:
1. No documents have been uploaded yet
2. The uploaded documents don't contain information about this topic
3. The question is outside the scope of the available data

Suggestions:
- Upload relevant documents if you haven't already
- Try rephrasing your question
- Ask about topics covered in your uploaded documents
"""

INSUFFICIENT_CONTEXT_PROMPT = """Based on the available documents, I found some potentially relevant information, but it's not sufficient to fully answer your question: "{query}"

What I found:
{context}

To get a better answer:
1. Upload more detailed documents on this topic
2. Rephrase your question to be more specific
3. Ask about aspects covered in the current documents
"""


# ============================================================================
# Helper Functions
# ============================================================================


def format_prompt(template: str, **kwargs: Any) -> str:
    """
    Format a prompt template with provided variables.

    Args:
        template: Prompt template string
        **kwargs: Variables to format into template

    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs)


def get_prompt_for_query_type(query_type: str) -> str:
    """
    Get the appropriate response prompt based on query type.

    Args:
        query_type: Type of query (SIMPLE, COMPLEX, ANALYTICAL)

    Returns:
        Appropriate prompt template
    """
    prompts = {
        "SIMPLE": QA_RESPONSE_PROMPT,
        "COMPLEX": QA_RESPONSE_PROMPT,
        "ANALYTICAL": ANALYTICAL_RESPONSE_PROMPT,
    }
    return prompts.get(query_type, QA_RESPONSE_PROMPT)


# Export all prompts
__all__ = [
    "SYSTEM_PROMPT",
    "QUERY_CLASSIFICATION_PROMPT",
    "QUERY_EXPANSION_PROMPT",
    "QA_RESPONSE_PROMPT",
    "ANALYTICAL_RESPONSE_PROMPT",
    "MULTI_STEP_ANALYSIS_PROMPT",
    "DOCUMENT_SUMMARY_PROMPT",
    "METADATA_EXTRACTION_PROMPT",
    "NO_CONTEXT_PROMPT",
    "INSUFFICIENT_CONTEXT_PROMPT",
    "format_prompt",
    "get_prompt_for_query_type",
]
