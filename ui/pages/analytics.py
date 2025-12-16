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
    
    st.divider()
    
    # Research Analysis Section
    render_research_analysis_section()


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


def render_research_analysis_section():
    """Render research analysis capabilities section."""
    st.subheader("🔬 Research Analysis Tools")
    st.markdown(
        "Advanced analysis tools for comparing papers, identifying trends, "
        "finding research gaps, and detecting consensus."
    )
    
    # Analysis mode selector
    analysis_mode = st.selectbox(
        "Analysis Type:",
        options=[
            "COMPARISON",
            "TREND_ANALYSIS",
            "GAP_IDENTIFICATION",
            "CONSENSUS_DETECTION",
        ],
        format_func=lambda x: {
            "COMPARISON": "📊 Comparison Matrix",
            "TREND_ANALYSIS": "📈 Trend Analysis",
            "GAP_IDENTIFICATION": "🔍 Research Gaps",
            "CONSENSUS_DETECTION": "⚖️ Consensus vs Controversy",
        }.get(x, x),
        help="Select the type of research analysis to perform",
    )
    
    # Show description based on mode
    descriptions = {
        "COMPARISON": "Compare multiple research approaches, methods, or papers side-by-side with customizable criteria.",
        "TREND_ANALYSIS": "Analyze how research topics or methods evolved over time.",
        "GAP_IDENTIFICATION": "Identify under-explored areas and research gaps in your corpus.",
        "CONSENSUS_DETECTION": "Detect what findings are widely accepted vs controversial.",
    }
    st.info(descriptions.get(analysis_mode, ""))
    
    # Configuration based on analysis mode
    config = {}
    
    if analysis_mode == "COMPARISON":
        st.markdown("**Comparison Criteria:**")
        col1, col2 = st.columns(2)
        
        with col1:
            criteria = st.multiselect(
                "Select criteria to compare:",
                options=["accuracy", "speed", "dataset", "year", "method_type", "parameters", "training_time"],
                default=["accuracy", "speed", "dataset", "year"],
                help="Select which aspects to compare across papers",
            )
        
        with col2:
            custom_criteria = st.text_input(
                "Additional criteria (comma-separated):",
                placeholder="e.g., memory_usage, inference_cost",
                help="Add custom criteria not in the default list",
            )
        
        # Combine criteria
        if custom_criteria:
            criteria.extend([c.strip() for c in custom_criteria.split(",") if c.strip()])
        
        config["criteria"] = criteria
    
    elif analysis_mode == "TREND_ANALYSIS":
        st.markdown("**Temporal Range:**")
        col1, col2 = st.columns(2)
        
        with col1:
            start_year = st.number_input(
                "Start Year:",
                min_value=2000,
                max_value=2024,
                value=2017,
                step=1,
            )
        
        with col2:
            end_year = st.number_input(
                "End Year:",
                min_value=2000,
                max_value=2024,
                value=2024,
                step=1,
            )
        
        config["time_range"] = (start_year, end_year)
        
        focus_area = st.text_input(
            "Focus Topic/Method (optional):",
            placeholder="e.g., attention mechanisms, transformers",
            help="Leave empty to analyze all topics",
        )
        if focus_area:
            config["focus_area"] = focus_area
    
    # Query input
    st.markdown("**Analysis Query:**")
    query_examples = {
        "COMPARISON": "So sánh BERT, GPT-2, và T5 về accuracy và speed",
        "TREND_ANALYSIS": "Các phương pháp attention phát triển như thế nào từ 2017-2024?",
        "GAP_IDENTIFICATION": "Vấn đề nào trong NLP chưa được nghiên cứu kỹ?",
        "CONSENSUS_DETECTION": "Pre-training có luôn cải thiện performance không?",
    }
    
    analysis_query = st.text_area(
        "Enter your research analysis query:",
        placeholder=query_examples.get(analysis_mode, ""),
        help="Describe what you want to analyze",
        height=100,
    )
    
    # Action buttons
    col1, col2 = st.columns([1, 4])
    
    with col1:
        run_analysis = st.button(
            "▶️ Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=not analysis_query,
        )
    
    with col2:
        if analysis_query:
            st.caption(f"Analysis will use **{analysis_mode}** mode")
        else:
            st.caption("Enter a query to run analysis")
    
    # Run analysis when button clicked
    if run_analysis and analysis_query:
        with st.spinner(f"Running {analysis_mode} analysis..."):
            try:
                # Import necessary components
                from src.retrieval.hybrid_retriever import default_hybrid_retriever
                from src.generation.research_analyzer import default_research_analyzer
                
                # Retrieve relevant documents
                st.info("🔍 Retrieving relevant documents...")
                retrieval_result = default_hybrid_retriever.search(
                    query=analysis_query,
                    top_k=20,  # Get more documents for analysis
                    search_method="hybrid",
                )
                
                if not retrieval_result or not retrieval_result.get("results"):
                    st.warning("No relevant documents found. Please upload papers related to your query first.")
                    return
                
                documents = retrieval_result["results"]
                st.success(f"Found {len(documents)} relevant documents")
                
                # Run analysis
                st.info(f"🔬 Performing {analysis_mode} analysis...")
                analysis_result = default_research_analyzer.analyze(
                    query=analysis_query,
                    documents=documents,
                    analysis_type=analysis_mode,
                    **config,
                )
                
                # Display results
                if "error" in analysis_result:
                    st.error(f"Analysis failed: {analysis_result['error']}")
                else:
                    st.success("✅ Analysis complete!")
                    
                    # Get the main analysis content
                    analysis_key_map = {
                        "COMPARISON": "comparison_matrix",
                        "TREND_ANALYSIS": "trend_analysis",
                        "GAP_IDENTIFICATION": "gap_analysis",
                        "CONSENSUS_DETECTION": "consensus_analysis",
                    }
                    
                    analysis_key = analysis_key_map.get(analysis_mode, "analysis")
                    analysis_content = analysis_result.get(analysis_key, "No analysis output")
                    
                    # Display main analysis
                    st.markdown("### 📋 Analysis Results")
                    st.markdown(analysis_content)
                    
                    # Display additional metadata if available
                    with st.expander("📊 View Analysis Metadata", expanded=False):
                        st.json(analysis_result)
                
            except ImportError as e:
                st.error(f"Import error: {e}")
                st.info("Make sure all required modules are installed.")
            except Exception as e:
                logger.error(f"Research analysis failed: {e}", exc_info=True)
                st.error(f"Analysis failed: {str(e)}")
                st.info("Try rephrasing your query or checking if relevant papers are uploaded.")


# Export for Streamlit pages
if __name__ == "__main__":
    render_analytics_dashboard()


__all__ = ["render_analytics_dashboard"]
