"""
Metrics Collection and Tracking for RAG System.

This module provides:
- Query metrics tracking (latency, tokens, cost)
- Performance metrics
- Usage analytics
- Persistent storage using SQLite
"""

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any

from src.core.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class MetricsCollector(LoggerMixin):
    """
    Collects and stores system metrics.
    
    Uses SQLite for persistent storage of:
    - Query metrics
    - Performance data
    - Usage statistics
    """
    
    def __init__(self, db_path: str | Path = "./data/metrics.db"):
        """
        Initialize metrics collector.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self.logger.info(f"MetricsCollector initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    query TEXT NOT NULL,
                    query_hash TEXT,
                    latency_ms REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    cost REAL,
                    query_type TEXT,
                    result_count INTEGER,
                    search_method TEXT,
                    success BOOLEAN DEFAULT TRUE,
                    error TEXT
                )
            """)
            
            # Cache metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cache_type TEXT,
                    hit BOOLEAN,
                    key_hash TEXT
                )
            """)
            
            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata TEXT
                )
            """)
            
            conn.commit()
    
    def record_query(
        self,
        query: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        query_type: str = "SIMPLE",
        result_count: int = 0,
        search_method: str = "vector",
        success: bool = True,
        error: str | None = None,
    ) -> int:
        """
        Record query metrics.
        
        Args:
            query: User query
            latency_ms: Response time in milliseconds
            prompt_tokens: Input tokens used
            completion_tokens: Output tokens generated
            cost: Estimated cost in USD
            query_type: Type of query
            result_count: Number of results returned
            search_method: Search method used (vector, hybrid, etc.)
            success: Whether query succeeded
            error: Error message if failed
            
        Returns:
            Record ID
        """
        query_hash = self._hash_text(query)
        total_tokens = prompt_tokens + completion_tokens
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO query_metrics 
                (query, query_hash, latency_ms, prompt_tokens, completion_tokens,
                 total_tokens, cost, query_type, result_count, search_method, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query[:500],  # Truncate long queries
                query_hash,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost,
                query_type,
                result_count,
                search_method,
                success,
                error,
            ))
            
            conn.commit()
            record_id = cursor.lastrowid
        
        self.logger.debug(f"Recorded query metric: {record_id}")
        return record_id
    
    def record_cache_event(
        self,
        cache_type: str,
        hit: bool,
        key: str = "",
    ):
        """
        Record cache hit/miss event.
        
        Args:
            cache_type: Type of cache (query, embedding, retrieval)
            hit: Whether it was a cache hit
            key: Cache key (will be hashed)
        """
        key_hash = self._hash_text(key) if key else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO cache_metrics (cache_type, hit, key_hash)
                VALUES (?, ?, ?)
            """, (cache_type, hit, key_hash))
            
            conn.commit()
    
    def record_metric(
        self,
        name: str,
        value: float,
        metadata: dict | None = None,
    ):
        """
        Record a generic system metric.
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_metrics (metric_name, metric_value, metadata)
                VALUES (?, ?, ?)
            """, (name, value, metadata_json))
            
            conn.commit()
    
    def get_summary(
        self,
        period: str = "day",
        start_date: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get metrics summary for a period.
        
        Args:
            period: Time period (hour, day, week, month, all)
            start_date: Optional start date
            
        Returns:
            Summary dictionary
        """
        if start_date is None:
            start_date = self._get_period_start(period)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query metrics summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(latency_ms) as avg_latency_ms,
                    MAX(latency_ms) as max_latency_ms,
                    MIN(latency_ms) as min_latency_ms,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost) as total_cost,
                    AVG(result_count) as avg_results,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_queries
                FROM query_metrics
                WHERE timestamp >= ?
            """, (start_date.isoformat(),))
            
            row = cursor.fetchone()
            
            # Cache hit rate
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN hit THEN 1 ELSE 0 END) as hits
                FROM cache_metrics
                WHERE timestamp >= ?
            """, (start_date.isoformat(),))
            
            cache_row = cursor.fetchone()
            cache_hit_rate = (cache_row[1] / cache_row[0] * 100) if cache_row[0] > 0 else 0
            
            return {
                "period": period,
                "start_date": start_date.isoformat(),
                "total_queries": row[0] or 0,
                "avg_latency_ms": round(row[1] or 0, 2),
                "max_latency_ms": round(row[2] or 0, 2),
                "min_latency_ms": round(row[3] or 0, 2),
                "total_tokens": row[4] or 0,
                "total_cost": round(row[5] or 0, 6),
                "avg_results": round(row[6] or 0, 1),
                "success_rate": round((row[7] / row[0] * 100) if row[0] > 0 else 100, 1),
                "cache_hit_rate": round(cache_hit_rate, 1),
            }
    
    def get_top_queries(self, limit: int = 10) -> list[dict]:
        """
        Get most common queries.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of popular queries with counts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT query_hash, query, COUNT(*) as count, AVG(latency_ms) as avg_latency
                FROM query_metrics
                GROUP BY query_hash
                ORDER BY count DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            return [
                {
                    "query": row[1][:100],  # Truncate for display
                    "count": row[2],
                    "avg_latency_ms": round(row[3], 2),
                }
                for row in rows
            ]
    
    def get_hourly_stats(self, hours: int = 24) -> list[dict]:
        """
        Get hourly statistics.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of hourly stats
        """
        start_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    strftime('%Y-%m-%d %H:00', timestamp) as hour,
                    COUNT(*) as queries,
                    AVG(latency_ms) as avg_latency,
                    SUM(cost) as cost
                FROM query_metrics
                WHERE timestamp >= ?
                GROUP BY hour
                ORDER BY hour
            """, (start_time.isoformat(),))
            
            rows = cursor.fetchall()
            
            return [
                {
                    "hour": row[0],
                    "queries": row[1],
                    "avg_latency_ms": round(row[2], 2),
                    "cost": round(row[3], 6),
                }
                for row in rows
            ]
    
    def get_search_method_stats(self) -> dict[str, int]:
        """Get query counts by search method."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT search_method, COUNT(*) as count
                FROM query_metrics
                GROUP BY search_method
            """)
            
            return {row[0] or "unknown": row[1] for row in cursor.fetchall()}
    
    def _get_period_start(self, period: str) -> datetime.datetime:
        """Get start datetime for a period."""
        now = datetime.datetime.now()
        
        if period == "hour":
            return now - datetime.timedelta(hours=1)
        elif period == "day":
            return now - datetime.timedelta(days=1)
        elif period == "week":
            return now - datetime.timedelta(weeks=1)
        elif period == "month":
            return now - datetime.timedelta(days=30)
        else:  # all
            return datetime.datetime(2000, 1, 1)
    
    def _hash_text(self, text: str) -> str:
        """Create a short hash of text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def clear_old_metrics(self, days: int = 30):
        """
        Clear metrics older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for table in ["query_metrics", "cache_metrics", "system_metrics"]:
                cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE timestamp < ?
                """, (cutoff.isoformat(),))
            
            conn.commit()
            
        self.logger.info(f"Cleared metrics older than {days} days")


# Default metrics collector instance
default_metrics_collector = MetricsCollector()


__all__ = [
    "MetricsCollector",
    "default_metrics_collector",
]
