"""
Chart Generator for data visualization.

This module generates Plotly charts from LLM responses
and structured data for visual data analysis.
"""

import json
import re
from typing import Any

from src.core.logging import LoggerMixin


class ChartGenerator(LoggerMixin):
    """
    Generates interactive charts using Plotly.
    
    Supports:
    - Line charts (trends, time series)
    - Bar charts (comparisons, categories)
    - Pie charts (distributions)
    - Scatter plots (correlations)
    """
    
    CHART_TYPES = ["line", "bar", "pie", "scatter", "auto"]
    
    def __init__(self):
        """Initialize chart generator."""
        self.logger.info("ChartGenerator initialized")
    
    def detect_chart_type(self, data: dict[str, Any]) -> str:
        """
        Auto-detect appropriate chart type based on data structure.
        
        Args:
            data: Data dictionary with 'x', 'y', 'labels', etc.
            
        Returns:
            Recommended chart type
        """
        x_data = data.get("x", [])
        y_data = data.get("y", [])
        labels = data.get("labels", [])
        values = data.get("values", [])
        
        # Check for time series (dates in x)
        if x_data and self._looks_like_dates(x_data):
            return "line"
        
        # Check for distribution (labels + values)
        if labels and values and not x_data:
            if len(labels) <= 8:
                return "pie"
            return "bar"
        
        # Check for comparison (categorical x)
        if x_data and y_data:
            if all(isinstance(x, str) for x in x_data):
                return "bar"
            if len(x_data) > 20:
                return "line"
            return "scatter"
        
        return "bar"  # Default
    
    def _looks_like_dates(self, data: list) -> bool:
        """Check if data looks like dates."""
        if not data:
            return False
        
        sample = str(data[0])
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'Q[1-4]\s*\d{4}',     # Q1 2024
            r'\d{4}',              # Year only
        ]
        
        return any(re.match(pattern, sample) for pattern in date_patterns)
    
    def generate_chart(
        self,
        data: dict[str, Any],
        chart_type: str = "auto",
        title: str = "",
        x_label: str = "",
        y_label: str = "",
    ) -> str:
        """
        Generate a Plotly chart as HTML.
        
        Args:
            data: Chart data with keys like 'x', 'y', 'labels', 'values'
            chart_type: Type of chart (line, bar, pie, scatter, auto)
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            
        Returns:
            HTML string with embedded Plotly chart
        """
        try:
            import plotly.graph_objects as go
            from plotly.utils import PlotlyJSONEncoder
        except ImportError:
            self.logger.error("Plotly not installed. Run: pip install plotly")
            return "<p>Chart generation requires Plotly. Install with: pip install plotly</p>"
        
        if chart_type == "auto":
            chart_type = self.detect_chart_type(data)
        
        self.logger.info(f"Generating {chart_type} chart")
        
        try:
            fig = self._create_figure(data, chart_type, title, x_label, y_label)
            
            # Convert to HTML
            html = fig.to_html(
                include_plotlyjs="cdn",
                full_html=False,
                config={"responsive": True, "displayModeBar": True}
            )
            
            return html
            
        except Exception as e:
            self.logger.error(f"Chart generation failed: {e}")
            return f"<p>Failed to generate chart: {e}</p>"
    
    def _create_figure(
        self,
        data: dict[str, Any],
        chart_type: str,
        title: str,
        x_label: str,
        y_label: str,
    ):
        """Create Plotly figure based on chart type."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        x = data.get("x", [])
        y = data.get("y", [])
        labels = data.get("labels", x)
        values = data.get("values", y)
        
        if chart_type == "line":
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=data.get("name", "Data"),
                line=dict(color="#0066cc", width=2),
                marker=dict(size=6),
            ))
            
        elif chart_type == "bar":
            fig.add_trace(go.Bar(
                x=labels if labels else x,
                y=values if values else y,
                name=data.get("name", "Data"),
                marker_color="#4CAF50",
            ))
            
        elif chart_type == "pie":
            fig.add_trace(go.Pie(
                labels=labels,
                values=values,
                hole=0.3,  # Donut style
                textposition="inside",
                textinfo="percent+label",
            ))
            
        elif chart_type == "scatter":
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=data.get("name", "Data"),
                marker=dict(
                    size=10,
                    color="#ff6b6b",
                    opacity=0.7,
                ),
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
        )
        
        return fig
    
    def extract_data_from_response(self, response: str) -> dict[str, Any] | None:
        """
        Extract chart-compatible data from LLM response.
        
        Looks for:
        - Markdown tables
        - JSON data blocks
        - Key-value patterns
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted data dictionary or None
        """
        # Try JSON extraction
        json_data = self._extract_json(response)
        if json_data:
            return json_data
        
        # Try table extraction
        table_data = self._extract_table(response)
        if table_data:
            return table_data
        
        return None
    
    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON data from text."""
        # Look for JSON code blocks
        json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        match = re.search(json_pattern, text)
        
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for inline JSON
        inline_pattern = r'\{[^{}]*"(?:x|y|labels|values)"[^{}]*\}'
        match = re.search(inline_pattern, text)
        
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _extract_table(self, text: str) -> dict | None:
        """Extract data from Markdown table."""
        # Find markdown table
        table_pattern = r'\|(.+)\|\n\|[-:\s|]+\|\n((?:\|.+\|\n?)+)'
        match = re.search(table_pattern, text)
        
        if not match:
            return None
        
        try:
            # Parse header
            header = [col.strip() for col in match.group(1).split('|') if col.strip()]
            
            # Parse rows
            rows = []
            for line in match.group(2).strip().split('\n'):
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    rows.append(cells)
            
            if len(header) >= 2 and rows:
                # Convert to chart data
                x = [row[0] for row in rows if row]
                y = []
                for row in rows:
                    if len(row) > 1:
                        try:
                            y.append(float(row[1].replace(',', '').replace('$', '')))
                        except ValueError:
                            y.append(0)
                
                return {
                    "labels": x,
                    "values": y,
                    "x": x,
                    "y": y,
                }
        except Exception as e:
            self.logger.debug(f"Table extraction failed: {e}")
        
        return None


# Default chart generator instance
default_chart_generator = ChartGenerator()


__all__ = [
    "ChartGenerator",
    "default_chart_generator",
]
