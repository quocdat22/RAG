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
1. GENERAL - General questions that can be answered from the documents
2. COMPARISON - Questions asking to compare multiple approaches, methods, or papers
3. TREND_ANALYSIS - Questions about temporal evolution, development over time, or research trends
4. GAP_IDENTIFICATION - Questions about missing research, under-explored areas, or research gaps
5. CONSENSUS_DETECTION - Questions about agreement/disagreement, consensus, or controversy in findings

User Query: {query}

Examples:
- "What is the transformer architecture?" → GENERAL
- "So sánh phương pháp A và B" → COMPARISON
- "Các phương pháp attention phát triển như thế nào từ 2017-2024?" → TREND_ANALYSIS
- "Vấn đề nào chưa được nghiên cứu kỹ?" → GAP_IDENTIFICATION
- "Pre-training có luôn cải thiện performance không?" → CONSENSUS_DETECTION

Respond with ONLY the category name.
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

# ============================================================================
# Research Analysis Prompts
# ============================================================================

COMPARISON_MATRIX_PROMPT = """Create a comprehensive comparison matrix for the research approaches mentioned in the query.

Query: {query}

Structured Data Extracted from Papers:
{structured_data}

Comparison Criteria: {criteria}

Instructions:
1. Create a **comparison table** with approaches as rows and criteria as columns
2. Fill in the table with specific values/findings from the papers
3. Use "N/A" for missing information
4. After the table, provide a **summary analysis** highlighting:
   - Key differences between approaches
   - Strengths and weaknesses of each
   - Which approach is best for what scenarios
5. Cite sources using [source_id] for each data point

Format your response as:

## Comparison Matrix

| Approach | Accuracy | Speed | Dataset | Year | Notes |
|----------|----------|-------|---------|------|-------|
| Method A | ...      | ...   | ...     | ...  | ...   |
| Method B | ...      | ...   | ...     | ...  | ...   |

## Analysis

[Your detailed comparison analysis here]

## Recommendations

[Which approach to use when]
"""

TREND_ANALYSIS_PROMPT = """Analyze the temporal evolution and trends in research based on the provided data.

Query: {query}

Temporal Data (sorted by year):
{temporal_data}

Time Range: {time_range}

Instructions:
1. Identify **key milestones** and breakthrough papers
2. Describe **evolutionary patterns**:
   - What changed over time?
   - What remained constant?
   - What new approaches emerged?
3. Identify **trends**:
   - Performance improvements
   - Methodological shifts
   - Dataset evolution
4. **Predict future directions** based on observed trends
5. Cite papers using [source_id]

Format your response as:

## Timeline of Key Developments

[Chronological narrative of major advances]

## Observed Trends

1. **Trend 1**: [Description]
2. **Trend 2**: [Description]
...

## Future Directions

[Predicted developments based on trends]
"""

GAP_IDENTIFICATION_PROMPT = """Identify research gaps and under-explored areas based on the corpus analysis.

Query: {query}

Corpus Coverage Summary:
{corpus_coverage}

Instructions:
1. Analyze what topics/methods/datasets are **well-covered** in the corpus
2. Identify what is **missing or under-explored**:
   - Methodological gaps
   - Dataset gaps  
   - Application domain gaps
   - Evaluation metric gaps
3. For each gap, explain:
   - Why it's important
   - What research would fill it
   - Potential impact
4. **Rank gaps by priority** (high/medium/low)
5. Cite papers to show coverage using [source_id]

Format your response as:

## Well-Covered Areas

[What the corpus thoroughly addresses]

## Identified Research Gaps

### High Priority Gaps
1. **Gap Name**: [Description]
   - Why important: ...
   - Suggested research: ...
   - Potential impact: ...

### Medium Priority Gaps
...

### Low Priority Gaps
...

## Summary

[Overall assessment of gaps]
"""

CONSENSUS_DETECTION_PROMPT = """Analyze consensus and controversy in research findings across multiple papers.

Query: {query}

Claims and Findings from Papers:
{claims}

Instructions:
1. Identify **widely agreed-upon findings** (consensus):
   - What do most/all papers agree on?
   - What evidence supports this consensus?
2. Identify **controversial or debated points**:
   - What do papers disagree on?
   - What are the conflicting findings?
   - What might explain the disagreement?
3. For each point, indicate:
   - **Consensus level**: Strong consensus / Moderate consensus / No consensus / Controversy
   - **Papers supporting**: [source_ids]
   - **Papers opposing**: [source_ids]
4. Provide **meta-analysis**: What does the field agree on vs what needs more research?

Format your response as:

## Strong Consensus ✅

1. **Finding**: [Description]
   - Supporting papers: [1], [2], [3]
   - Evidence: ...

## Moderate Consensus 🟨

1. **Finding**: [Description]
   - Majority view: [papers]
   - Minority view: [papers]

## Active Controversies ⚠️

1. **Debate**: [Topic]
   - Position A: [Description] - Papers: [1], [2]
   - Position B: [Description] - Papers: [3], [4]
   - Explanation: ...

## Meta-Analysis

[Overall assessment of field agreement]
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
        query_type: Type of query (GENERAL, COMPARISON, etc.)

    Returns:
        Appropriate prompt template
    """
    prompts = {
        "GENERAL": QA_RESPONSE_PROMPT,
        "SIMPLE": QA_RESPONSE_PROMPT,
        "COMPLEX": QA_RESPONSE_PROMPT,
        "COMPARISON": COMPARISON_MATRIX_PROMPT,
        "TREND_ANALYSIS": TREND_ANALYSIS_PROMPT,
        "GAP_IDENTIFICATION": GAP_IDENTIFICATION_PROMPT,
        "CONSENSUS_DETECTION": CONSENSUS_DETECTION_PROMPT,
    }
    return prompts.get(query_type, QA_RESPONSE_PROMPT)


# Export all prompts
__all__ = [
    "SYSTEM_PROMPT",
    "QUERY_CLASSIFICATION_PROMPT",
    "QUERY_EXPANSION_PROMPT",
    "QA_RESPONSE_PROMPT",
    "DOCUMENT_SUMMARY_PROMPT",
    "METADATA_EXTRACTION_PROMPT",
    "NO_CONTEXT_PROMPT",
    "INSUFFICIENT_CONTEXT_PROMPT",
    "COMPARISON_MATRIX_PROMPT",
    "TREND_ANALYSIS_PROMPT",
    "GAP_IDENTIFICATION_PROMPT",
    "CONSENSUS_DETECTION_PROMPT",
    "format_prompt",
    "get_prompt_for_query_type",
]
