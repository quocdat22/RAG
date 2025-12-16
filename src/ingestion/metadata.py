"""
Metadata extraction from documents.

This module provides utilities for extracting and enriching document metadata,
with specialized support for academic papers including abstract, DOI, arXiv ID,
venue detection, institution extraction, and LLM-based deep analysis.
"""

import json
import re
from typing import Any

from langdetect import detect, LangDetectException

from src.core.logging import LoggerMixin
from src.ingestion.loaders import Document


class MetadataExtractor(LoggerMixin):
    """
    Extracts and enriches document metadata.
    
    Supports both regex-based extraction and LLM-based deep extraction
    for academic papers.
    """
    
    # Common academic conferences
    CONFERENCES = [
        "NeurIPS", "NIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV",
        "ACL", "EMNLP", "NAACL", "COLING", "AAAI", "IJCAI", "KDD",
        "WWW", "SIGIR", "CIKM", "WSDM", "ICDE", "VLDB", "SIGMOD",
        "FSE", "ICSE", "ASE", "ISSTA", "PLDI", "POPL", "OOPSLA",
        "CHI", "UIST", "CSCW", "UbiComp", "MobiCom", "SenSys",
        "OSDI", "SOSP", "NSDI", "EuroSys", "ATC", "FAST",
        "S&P", "CCS", "USENIX Security", "NDSS",
    ]
    
    # Common academic journals
    JOURNALS = [
        "Nature", "Science", "Cell", "PNAS", "Nature Machine Intelligence",
        "Nature Communications", "Scientific Reports", "JMLR", 
        "Journal of Machine Learning Research", "TACL", "TPAMI",
        "IEEE Transactions", "ACM Computing Surveys", "ACM TOCS",
        "Artificial Intelligence", "Machine Learning", "Neural Networks",
        "Pattern Recognition", "Computer Vision and Image Understanding",
    ]
    
    # Common institutions for academic papers
    INSTITUTION_PATTERNS = [
        r"University of \w+(?:\s+\w+)?",
        r"\w+ University",
        r"\w+ Institute of Technology",
        r"MIT", r"Stanford", r"Berkeley", r"CMU", r"Caltech",
        r"Google(?:\s+Research)?", r"DeepMind", r"OpenAI", r"Meta(?:\s+AI)?",
        r"Microsoft(?:\s+Research)?", r"Amazon(?:\s+AWS)?", r"Apple",
        r"NVIDIA", r"IBM(?:\s+Research)?", r"Facebook(?:\s+AI)?",
        r"Tsinghua", r"Peking University", r"ETH Zurich", r"EPFL",
        r"Oxford", r"Cambridge", r"Imperial College",
    ]

    def __init__(self, llm_client=None):
        """
        Initialize metadata extractor.
        
        Args:
            llm_client: Optional LLM client for deep extraction
        """
        self._llm_client = llm_client
    
    @property
    def llm_client(self):
        """Lazy-load LLM client if not provided."""
        if self._llm_client is None:
            from src.generation.llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client
    
    def extract_language(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            ISO 639-1 language code (e.g., 'en', 'vi')
        """
        try:
            # Use first 1000 chars for detection
            sample = text[:1000]
            lang = detect(sample)
            return lang
        except LangDetectException:
            self.logger.warning("Failed to detect language, defaulting to 'en'")
            return "en"

    def extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """
        Extract keywords from text.

        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords

        Returns:
            List of keywords
        """
        from collections import Counter

        # Convert to lowercase and split into words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Common stop words to filter
        stop_words = {
            "the", "is", "at", "which", "on", "and", "or", "but", "in", "with",
            "to", "for", "of", "as", "by", "that", "this", "from", "are", "was",
            "were", "been", "have", "has", "had", "will", "would", "can", "could",
            "should", "may", "might", "must", "shall", "our", "we", "they", "their",
            "these", "those", "such", "when", "where", "how", "what", "who", "why",
            "also", "than", "then", "more", "most", "some", "any", "all", "each",
            "both", "few", "many", "much", "other", "another", "same", "different",
        }

        # Filter stop words
        words = [w for w in words if w not in stop_words]

        # Count frequency
        word_counts = Counter(words)

        # Get top keywords
        keywords = [word for word, count in word_counts.most_common(max_keywords)]

        return keywords

    def extract_abstract(self, text: str) -> str:
        """
        Extract abstract from academic paper.
        
        Args:
            text: Document text
            
        Returns:
            Abstract text or empty string if not found
        """
        # Pattern 1: Look for "Abstract" section
        patterns = [
            r"(?:^|\n)\s*(?:ABSTRACT|Abstract)\s*\n(.*?)(?:\n\s*(?:1\.?\s*)?(?:INTRODUCTION|Introduction|KEYWORDS|Keywords|1\s+Introduction)|\n\n\n)",
            r"(?:^|\n)\s*(?:ABSTRACT|Abstract)[:\s]+(.*?)(?:\n\s*(?:INTRODUCTION|Introduction|KEYWORDS|Keywords)|\n\n)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                # Clean up abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:  # Minimum reasonable abstract length
                    return abstract[:2000]  # Cap at 2000 chars
        
        return ""
    
    def extract_doi(self, text: str) -> str:
        """
        Extract DOI from document.
        
        Args:
            text: Document text
            
        Returns:
            DOI string or empty if not found
        """
        # DOI pattern: 10.xxxx/xxxxx
        patterns = [
            r"(?:DOI|doi)[:\s]+([10]\.\d{4,}/[^\s]+)",
            r"https?://(?:dx\.)?doi\.org/([10]\.\d{4,}/[^\s]+)",
            r"(?:^|\s)(10\.\d{4,}/[^\s]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:3000])
            if match:
                doi = match.group(1).strip()
                # Clean trailing punctuation
                doi = re.sub(r'[,.\s]+$', '', doi)
                return doi
        
        return ""
    
    def extract_arxiv_id(self, text: str) -> str:
        """
        Extract arXiv ID from document.
        
        Args:
            text: Document text
            
        Returns:
            arXiv ID or empty if not found
        """
        patterns = [
            r"arXiv[:\s]+(\d{4}\.\d{4,}(?:v\d+)?)",
            r"arxiv\.org/abs/(\d{4}\.\d{4,}(?:v\d+)?)",
            r"arxiv\.org/pdf/(\d{4}\.\d{4,}(?:v\d+)?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    def extract_venue(self, text: str) -> dict[str, str]:
        """
        Detect conference or journal venue.
        
        Args:
            text: Document text
            
        Returns:
            Dict with 'type' (conference/journal) and 'name'
        """
        sample = text[:5000].upper()
        
        # Check conferences
        for conf in self.CONFERENCES:
            # Look for conference name with year
            pattern = rf"\b{conf.upper()}\s*'?\d{{2,4}}\b"
            match = re.search(pattern, sample)
            if match:
                return {"type": "conference", "name": match.group(0)}
            
            # Just the name
            if conf.upper() in sample:
                return {"type": "conference", "name": conf}
        
        # Check journals
        for journal in self.JOURNALS:
            if journal.upper() in sample:
                return {"type": "journal", "name": journal}
        
        return {"type": "", "name": ""}
    
    def extract_institutions(self, text: str) -> list[str]:
        """
        Extract author institutions/affiliations.
        
        Args:
            text: Document text
            
        Returns:
            List of institution names
        """
        institutions = set()
        
        # Use first 3000 chars (usually contains affiliations)
        sample = text[:3000]
        
        for pattern in self.INSTITUTION_PATTERNS:
            matches = re.findall(pattern, sample, re.IGNORECASE)
            for match in matches:
                if len(match) > 3:  # Filter too short matches
                    institutions.add(match.strip())
        
        return list(institutions)[:10]  # Limit to 10
    
    def extract_with_llm(self, text: str) -> dict[str, Any]:
        """
        Use LLM for deep metadata extraction.
        
        Extracts:
        - Main contribution summary
        - Methodology type
        - Research domain
        - Key findings
        
        Args:
            text: Document text
            
        Returns:
            Dict with extracted metadata
        """
        # Use first 4000 chars to stay within context limits
        sample = text[:4000]
        
        prompt = f"""Analyze this academic paper excerpt and extract metadata.

Paper Content:
{sample}

Extract the following information as a JSON object:
1. "contribution": A one-sentence summary of the main contribution (max 100 words)
2. "methodology": Type of methodology used (e.g., "empirical study", "theoretical analysis", "survey", "benchmark", "neural network", "transformer-based", etc.)
3. "research_domain": Primary research domain (e.g., "natural language processing", "computer vision", "machine learning", "robotics", etc.)
4. "key_findings": List of 2-3 key findings or results (short bullet points)
5. "datasets": List of datasets mentioned (if any)
6. "models": List of models/methods proposed or compared (if any)

Return ONLY valid JSON. Use empty string or empty list for missing fields.

JSON:"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt="You are a precise academic paper metadata extractor. Extract only factual information and return valid JSON.",
                temperature=0.1,
                max_tokens=500,
            )
            
            # Parse JSON response
            return self._parse_json_response(response)
            
        except Exception as e:
            self.logger.warning(f"LLM extraction failed: {e}")
            return {}
    
    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON: {e}")
            return {}

    def categorize_document(self, text: str, metadata: dict[str, Any]) -> str:
        """
        Categorize document based on scientific domain.

        Args:
            text: Document text
            metadata: Document metadata

        Returns:
            Category name (Scientific Domain)
        """
        text_lower = text.lower()

        # 1. Computer Science & AI
        cs_terms = [
            "algorithm", "neural network", "machine learning", "deep learning",
            "artificial intelligence", "software", "database", "cloud computing",
            "security", "network", "system", "data science", "large language model",
            "transformer", "attention mechanism", "backpropagation", "gradient",
        ]
        if any(term in text_lower for term in cs_terms) or "computer science" in text_lower:
            return "computer_science"

        # 2. Medicine & Biology
        bio_terms = [
            "patient", "clinical", "treatment", "disease", "cell", "protein",
            "gene", "genome", "medical", "biological", "hospital", "virus",
            "bacteria", "diagnosis", "therapy", "pharmaceutical",
        ]
        if any(term in text_lower for term in bio_terms):
            return "medicine_biology"

        # 3. Physics & Chemistry
        phys_terms = [
            "quantum", "particle", "energy", "magnetic", "velocity", "reaction",
            "chemical", "analyte", "synthesis", "atom", "thermodynamics",
            "electromagnetic", "relativity", "photon",
        ]
        if any(term in text_lower for term in phys_terms):
            return "physics_chemistry"

        # 4. Mathematics & Statistics
        math_terms = [
            "theorem", "lemma", "proof", "proposition", "corollary", "equation",
            "stochastic", "distribution", "variance", "algebra", "topology",
            "differential", "integral", "matrix",
        ]
        if any(term in text_lower for term in math_terms):
            return "mathematics_statistics"

        # 5. Economics & Business
        econ_terms = [
            "market", "economy", "finance", "supply chain", "consumer",
            "marketing", "stock", "inflation", "gdp", "investment",
        ]
        if any(term in text_lower for term in econ_terms):
            return "economics_business"

        # Default category
        return "general_science"

    def enrich_metadata(self, document: Document, use_llm: bool = True) -> Document:
        """
        Enrich document metadata with extracted information.

        Args:
            document: Document to enrich
            use_llm: Whether to use LLM for deep extraction (default True)

        Returns:
            Document with enriched metadata
        """
        try:
            self.logger.info(f"Enriching metadata for document: {document.doc_id}")

            content = document.content
            
            # Basic extraction (regex-based)
            # Language detection
            language = self.extract_language(content)
            document.metadata["language"] = language

            # Keywords
            keywords = self.extract_keywords(content)
            document.metadata["keywords"] = keywords

            # Category
            category = self.categorize_document(content, document.metadata)
            document.metadata["category"] = category

            # Content stats
            document.metadata["content_length"] = len(content)
            document.metadata["word_count"] = len(content.split())
            
            # Academic paper specific extraction
            # Abstract
            abstract = self.extract_abstract(content)
            if abstract:
                document.metadata["abstract"] = abstract
            
            # DOI
            doi = self.extract_doi(content)
            if doi:
                document.metadata["doi"] = doi
            
            # arXiv ID
            arxiv_id = self.extract_arxiv_id(content)
            if arxiv_id:
                document.metadata["arxiv_id"] = arxiv_id
            
            # Venue (conference/journal)
            venue = self.extract_venue(content)
            if venue["name"]:
                document.metadata["venue"] = venue
            
            # Institutions
            institutions = self.extract_institutions(content)
            if institutions:
                document.metadata["institutions"] = institutions
            
            # LLM-based deep extraction
            if use_llm:
                try:
                    llm_metadata = self.extract_with_llm(content)
                    if llm_metadata:
                        document.metadata["llm_extracted"] = llm_metadata
                        
                        # Promote key fields to top level
                        if llm_metadata.get("contribution"):
                            document.metadata["contribution"] = llm_metadata["contribution"]
                        if llm_metadata.get("methodology"):
                            document.metadata["methodology"] = llm_metadata["methodology"]
                        if llm_metadata.get("research_domain"):
                            document.metadata["research_domain"] = llm_metadata["research_domain"]
                        
                except Exception as e:
                    self.logger.warning(f"LLM extraction skipped: {e}")

            self.logger.info(
                f"Enriched metadata: lang={language}, category={category}, "
                f"abstract={'yes' if abstract else 'no'}, doi={doi or 'N/A'}"
            )

            return document

        except Exception as e:
            self.logger.error(f"Failed to enrich metadata: {e}")
            # Return document unchanged if enrichment fails
            return document


# Default extractor instance (without LLM client - will lazy load)
default_extractor = MetadataExtractor()


def enrich_document_metadata(document: Document, use_llm: bool = True) -> Document:
    """
    Enrich document metadata using default extractor.

    Args:
        document: Document to enrich
        use_llm: Whether to use LLM for deep extraction

    Returns:
        Document with enriched metadata
    """
    return default_extractor.enrich_metadata(document, use_llm=use_llm)


__all__ = [
    "MetadataExtractor",
    "enrich_document_metadata",
    "default_extractor",
]

