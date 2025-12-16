"""
Prompt Templates for RAG System

This module contains all LLM prompt templates used throughout the system.
Templates are organized by use case and follow best practices for prompt engineering.
"""

from typing import Any


# ============================================================================
# System Prompts
# ============================================================================

SYSTEM_PROMPT = """You are an expert research assistant AI specialized in helping scientists and researchers find relevant information from academic papers, surveys, and preprints.

Guidelines:
1. Base your answers ONLY on the provided papers - never make up information
2. Always cite the **paper title** using format: [Paper Title] or [filename] if title unavailable
3. Include **original excerpts/quotes** from papers to support your answers
4. If the information is not in the papers, clearly state "I couldn't find relevant information in the available papers."
5. Provide specific findings, methods, results, and conclusions when available
6. Be concise but comprehensive
7. Maintain a helpful, academic tone

Your responses should:
- Start with a direct answer to the question
- Include relevant excerpts from papers with citations
- List the source papers at the end
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

QA_RESPONSE_PROMPT = """Based on the academic papers below, answer the researcher's question.

Papers:
{context}

Paper Information:
{metadata}

Research Question:
{query}

Instructions:
1. Answer based ONLY on the provided papers
2. For each key point, include an **original excerpt** from the paper in quotes
3. Use **numbered citations** like [1], [2], etc. to reference papers
4. **Preserve mathematical formulas** exactly as written (e.g., LaTeX notation)
5. If relevant info not found, state "I couldn't find this information in the available papers."

Format your response as:
1. Direct answer with supporting excerpts and numbered citations [1], [2]
2. Each excerpt should be quoted: "..." [1]
3. At the END, add a "**References:**" section with FULL ACADEMIC CITATIONS:
   Format: [1] Author et al., Year, Paper title, Section, p. X
   
   Examples:
   - [1] Vaswani et al., 2017, Attention Is All You Need, Introduction, p. 2
   - [2] Brown et al., 2020, Language Models are Few-Shot Learners, Methods, p. 15
   
   If any field is missing:
   - No author: use "Unknown Author"
   - No year: use "n.d."
   - No section: use the chunk position info
   - Page is always estimated from the chunk position

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

NO_CONTEXT_PROMPT = """I couldn't find any relevant papers to answer your question: "{query}"

This could be because:
1. No papers have been uploaded yet
2. The available papers don't contain information about this topic
3. The question is outside the scope of the uploaded papers

Suggestions:
- Upload relevant research papers if you haven't already
- Try rephrasing your question with different keywords
- Ask about topics covered in your uploaded papers
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
