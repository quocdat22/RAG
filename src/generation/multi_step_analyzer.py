"""
Multi-step Analyzer for complex analytical queries.

This module implements Chain-of-Thought reasoning for
complex analysis tasks that require multiple steps.
"""

from typing import Any

from config import prompts
from src.core.logging import LoggerMixin, log_execution_time
from src.generation.llm_client import LLMClient


class MultiStepAnalyzer(LoggerMixin):
    """
    Performs multi-step analysis using Chain-of-Thought.
    
    Steps:
    1. Data extraction from retrieved documents
    2. Statistical analysis and calculations
    3. Pattern detection and insights
    4. Recommendations generation
    5. Final synthesis and report
    """
    
    ANALYSIS_STEPS = [
        "data_extraction",
        "statistical_analysis", 
        "pattern_detection",
        "recommendations",
        "synthesis",
    ]
    
    def __init__(self, llm_client: LLMClient | None = None):
        """
        Initialize multi-step analyzer.
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm_client = llm_client or LLMClient()
        self.logger.info("MultiStepAnalyzer initialized")
    
    @log_execution_time
    def analyze(
        self,
        query: str,
        documents: list[dict[str, Any]],
        include_steps: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Perform multi-step analysis on documents.
        
        Args:
            query: Analysis query
            documents: Retrieved documents
            include_steps: Optional list of steps to include
            
        Returns:
            Analysis result with steps and final report
        """
        steps = include_steps or self.ANALYSIS_STEPS
        
        self.logger.info(f"Starting multi-step analysis with {len(steps)} steps")
        
        # Build document context
        context = self._build_context(documents)
        
        # Execute each step
        results = {
            "query": query,
            "steps": {},
            "final_report": "",
            "metadata": {
                "document_count": len(documents),
                "steps_executed": [],
            }
        }
        
        accumulated_insights = ""
        
        for step in steps:
            self.logger.info(f"Executing step: {step}")
            
            try:
                step_result = self._execute_step(
                    step=step,
                    query=query,
                    context=context,
                    previous_insights=accumulated_insights,
                )
                
                results["steps"][step] = {
                    "content": step_result,
                    "status": "completed",
                }
                results["metadata"]["steps_executed"].append(step)
                
                # Accumulate insights for next step
                accumulated_insights += f"\n\n## {step.replace('_', ' ').title()}\n{step_result}"
                
            except Exception as e:
                self.logger.error(f"Step {step} failed: {e}")
                results["steps"][step] = {
                    "content": f"Error: {str(e)}",
                    "status": "failed",
                }
        
        # Generate final synthesis
        if "synthesis" in results["steps"]:
            results["final_report"] = results["steps"]["synthesis"]["content"]
        else:
            results["final_report"] = accumulated_insights
        
        self.logger.info("Multi-step analysis completed")
        
        return results
    
    def _build_context(self, documents: list[dict[str, Any]]) -> str:
        """Build context from documents."""
        parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("document", "")
            source = doc.get("metadata", {}).get("filename", f"Document {i}")
            parts.append(f"[Source: {source}]\n{content}")
        
        return "\n\n---\n\n".join(parts)
    
    def _execute_step(
        self,
        step: str,
        query: str,
        context: str,
        previous_insights: str,
    ) -> str:
        """Execute a single analysis step."""
        
        step_prompts = {
            "data_extraction": """
Based on the following documents, extract all relevant data points, numbers, 
facts, and figures that relate to the query.

Query: {query}

Documents:
{context}

List the key data points in a structured format.
""",
            "statistical_analysis": """
Analyze the following data and perform statistical analysis:
- Calculate totals, averages, percentages
- Identify trends (increasing, decreasing, stable)
- Note any significant outliers or anomalies

Query: {query}

Previous findings:
{previous_insights}

Provide statistical insights.
""",
            "pattern_detection": """
Based on the analysis so far, identify:
- Patterns and trends
- Correlations between data points
- Anomalies or unexpected findings
- Seasonal or cyclical patterns

Query: {query}

Previous analysis:
{previous_insights}

Describe detected patterns with evidence.
""",
            "recommendations": """
Based on the patterns and analysis, provide actionable recommendations:
- What actions should be taken?
- What opportunities exist?
- What risks should be mitigated?
- What further analysis is needed?

Query: {query}

Analysis results:
{previous_insights}

Provide specific, actionable recommendations.
""",
            "synthesis": """
Create a comprehensive executive summary that synthesizes all findings:

Query: {query}

Complete Analysis:
{previous_insights}

Format as a professional report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Recommendations (prioritized list)
4. Conclusion
""",
        }
        
        prompt_template = step_prompts.get(step, step_prompts["synthesis"])
        
        prompt = prompt_template.format(
            query=query,
            context=context,
            previous_insights=previous_insights,
        )
        
        return self.llm_client.generate(
            prompt=prompt,
            system_prompt="You are an expert data analyst performing systematic analysis.",
        )
    
    def generate_report_markdown(self, results: dict[str, Any]) -> str:
        """
        Generate a formatted Markdown report from analysis results.
        
        Args:
            results: Analysis results from analyze()
            
        Returns:
            Formatted Markdown report
        """
        report_parts = [
            f"# Analysis Report",
            f"\n**Query:** {results['query']}\n",
            f"**Documents Analyzed:** {results['metadata']['document_count']}",
            f"**Steps Completed:** {len(results['metadata']['steps_executed'])}\n",
            "---\n",
        ]
        
        # Add each step's content
        for step, data in results["steps"].items():
            step_title = step.replace("_", " ").title()
            status_icon = "✅" if data["status"] == "completed" else "❌"
            
            report_parts.append(f"## {status_icon} {step_title}\n")
            report_parts.append(data["content"])
            report_parts.append("\n---\n")
        
        return "\n".join(report_parts)


# Default analyzer instance
default_multi_step_analyzer = MultiStepAnalyzer()


__all__ = [
    "MultiStepAnalyzer",
    "default_multi_step_analyzer",
]
