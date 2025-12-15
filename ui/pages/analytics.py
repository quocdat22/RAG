"""
Analytics Dashboard for RAG System.

Provides visualization and analysis of:
- Query metrics (volume, latency, cost)
- Usage trends
- Top queries
- Performance statistics
"""

import streamlit as st

from src.core.logging import get_logger
from src.core.metrics import default_metrics_collector

logger = get_logger(__name__)


def render_analytics_dashboard():
    """Render the analytics dashboard page."""
    st.header("📊 Analytics Dashboard")
    st.markdown("Monitor system performance, usage, and costs.")
    
    # Period selector
    period = st.selectbox(
        "Time Period:",
        options=["hour", "day", "week", "month", "all"],
        index=1,
        format_func=lambda x: {
            "hour": "Last Hour",
            "day": "Last 24 Hours",
            "week": "Last Week",
            "month": "Last Month",
            "all": "All Time",
        }.get(x, x),
    )
    
    try:
        # Get summary metrics
        summary = default_metrics_collector.get_summary(period=period)
        
        # Top-level metrics
        st.subheader("📈 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Queries",
                summary["total_queries"],
                help="Number of queries in this period",
            )
        
        with col2:
            st.metric(
                "Avg Latency",
                f"{summary['avg_latency_ms']:.0f}ms",
                help="Average response time",
            )
        
        with col3:
            st.metric(
                "Total Cost",
                f"${summary['total_cost']:.4f}",
                help="Estimated API cost",
            )
        
        with col4:
            st.metric(
                "Success Rate",
                f"{summary['success_rate']:.1f}%",
                help="Percentage of successful queries",
            )
        
        st.divider()
        
        # Second row of metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric(
                "Total Tokens",
                f"{summary['total_tokens']:,}",
                help="Total tokens processed",
            )
        
        with col6:
            st.metric(
                "Avg Results",
                f"{summary['avg_results']:.1f}",
                help="Average documents per query",
            )
        
        with col7:
            st.metric(
                "Cache Hit Rate",
                f"{summary['cache_hit_rate']:.1f}%",
                help="Percentage of cache hits",
            )
        
        with col8:
            st.metric(
                "Max Latency",
                f"{summary['max_latency_ms']:.0f}ms",
                help="Maximum response time",
            )
        
        st.divider()
        
        # Charts section
        render_charts(period)
        
        st.divider()
        
        # Top queries section
        render_top_queries()
        
        st.divider()
        
        # Search method breakdown
        render_search_method_stats()
        
    except Exception as e:
        logger.error(f"Failed to load analytics: {e}")
        st.error(f"Failed to load analytics: {e}")
        st.info("Analytics data will be available after some queries are made.")


def render_charts(period: str):
    """Render analytics charts."""
    st.subheader("📉 Trends")
    
    try:
        # Get hourly stats
        hours = {
            "hour": 1,
            "day": 24,
            "week": 168,
            "month": 720,
            "all": 720,
        }.get(period, 24)
        
        hourly_stats = default_metrics_collector.get_hourly_stats(hours=min(hours, 168))
        
        if hourly_stats:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Query Volume", "Average Latency", "Cost", "Cumulative Cost"),
                vertical_spacing=0.15,
                horizontal_spacing=0.1,
            )
            
            hours = [s["hour"] for s in hourly_stats]
            queries = [s["queries"] for s in hourly_stats]
            latencies = [s["avg_latency_ms"] for s in hourly_stats]
            costs = [s["cost"] for s in hourly_stats]
            cumulative_costs = []
            total = 0
            for c in costs:
                total += c
                cumulative_costs.append(total)
            
            # Query volume
            fig.add_trace(
                go.Bar(x=hours, y=queries, name="Queries", marker_color="#4CAF50"),
                row=1, col=1
            )
            
            # Latency
            fig.add_trace(
                go.Scatter(x=hours, y=latencies, name="Latency (ms)", 
                          mode="lines+markers", line=dict(color="#2196F3")),
                row=1, col=2
            )
            
            # Cost per hour
            fig.add_trace(
                go.Bar(x=hours, y=costs, name="Cost ($)", marker_color="#FF9800"),
                row=2, col=1
            )
            
            # Cumulative cost
            fig.add_trace(
                go.Scatter(x=hours, y=cumulative_costs, name="Cumulative ($)",
                          mode="lines", fill="tozeroy", line=dict(color="#9C27B0")),
                row=2, col=2
            )
            
            fig.update_layout(
                height=500,
                showlegend=False,
                template="plotly_white",
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for charts. Start making queries to see trends!")
            
    except ImportError:
        st.warning("Plotly not installed. Charts disabled. Run: pip install plotly")
    except Exception as e:
        logger.warning(f"Failed to render charts: {e}")
        st.warning("Charts temporarily unavailable.")


def render_top_queries():
    """Render top queries section."""
    st.subheader("🔝 Top Queries")
    
    try:
        top_queries = default_metrics_collector.get_top_queries(limit=10)
        
        if top_queries:
            # Display as table
            import pandas as pd
            
            df = pd.DataFrame(top_queries)
            df.columns = ["Query", "Count", "Avg Latency (ms)"]
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No queries recorded yet.")
            
    except Exception as e:
        logger.warning(f"Failed to render top queries: {e}")


def render_search_method_stats():
    """Render search method breakdown."""
    st.subheader("🔍 Search Methods")
    
    try:
        method_stats = default_metrics_collector.get_search_method_stats()
        
        if method_stats:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                for method, count in method_stats.items():
                    st.metric(method.title(), count)
            
            with col2:
                try:
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=list(method_stats.keys()),
                        values=list(method_stats.values()),
                        hole=0.4,
                    )])
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=20, b=20),
                        showlegend=True,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    pass
        else:
            st.info("No search method data available yet.")
            
    except Exception as e:
        logger.warning(f"Failed to render search method stats: {e}")


# Export for Streamlit pages
if __name__ == "__main__":
    render_analytics_dashboard()


__all__ = ["render_analytics_dashboard"]
