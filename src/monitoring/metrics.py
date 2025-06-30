"""
Metrics collection and monitoring system for the e-commerce recommendation API.

This module provides comprehensive monitoring capabilities including:
- API performance metrics (response time, request count, error rate)
- Business metrics (CTR, conversion rate, catalog coverage)
- Model health metrics (prediction confidence, feature drift)
- Real-time metric collection and aggregation
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from pathlib import Path
from contextlib import contextmanager

# Настройка Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricPoint:
    """Individual metric data point."""
    
    def __init__(self, name: str, value: float, timestamp: datetime = None, tags: Dict[str, str] = None):
        self.name = name
        self.value = value
        self.timestamp = timestamp or datetime.now()
        self.tags = tags or {}


class APIMetrics:
    """API performance metrics."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.min_response_time = float('inf')
        self.max_response_time = 0.0
    
    @property
    def avg_response_time(self) -> float:
        return self.total_response_time / max(self.request_count, 1)
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.request_count, 1)


class BusinessMetrics:
    """Business-related metrics."""
    
    def __init__(self):
        self.recommendations_shown = 0
        self.recommendations_clicked = 0
        self.items_added_to_cart = 0
        self.unique_items_recommended = set()
        self.total_catalog_size = 0
    
    @property
    def ctr(self) -> float:
        """Click-through rate."""
        return self.recommendations_clicked / max(self.recommendations_shown, 1)
    
    @property
    def conversion_rate(self) -> float:
        """Conversion rate to cart additions."""
        return self.items_added_to_cart / max(self.recommendations_shown, 1)
    
    @property
    def catalog_coverage(self) -> float:
        """Percentage of catalog covered by recommendations."""
        return len(self.unique_items_recommended) / max(self.total_catalog_size, 1)


class MetricsCollector:
    """
    Central metrics collection system.
    
    Collects, aggregates, and exports metrics for monitoring dashboards.
    Thread-safe and supports real-time metric updates.
    """
    
    def __init__(self, export_interval: int = 60, max_history_size: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            export_interval: Interval in seconds for exporting metrics
            max_history_size: Maximum number of historical data points to keep
        """
        self.export_interval = export_interval
        self.max_history_size = max_history_size
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._metrics_history = deque(maxlen=max_history_size)
        self._current_window = defaultdict(list)
        
        # Current metrics
        self.api_metrics = APIMetrics()
        self.business_metrics = BusinessMetrics()
        
        # Эндпоинт-specific metrics
        self.endpoint_metrics = defaultdict(APIMetrics)
        
        # Export thread
        self._export_thread = None
        self._stop_export = threading.Event()
        
        # Запуск background export
        self.start_export_thread()
    
    def start_export_thread(self):
        """Start background thread for periodic metric export."""
        if self._export_thread is None or not self._export_thread.is_alive():
            self._export_thread = threading.Thread(
                target=self._export_loop,
                daemon=True
            )
            self._export_thread.start()
            logger.info("Metrics export thread started")
    
    def stop_export_thread(self):
        """Stop background export thread."""
        self._stop_export.set()
        if self._export_thread and self._export_thread.is_alive():
            self._export_thread.join(timeout=5)
            logger.info("Metrics export thread stopped")
    
    def _export_loop(self):
        """Background loop for exporting metrics."""
        while not self._stop_export.wait(self.export_interval):
            try:
                self.export_metrics()
            except Exception as e:
                logger.error(f"Error exporting metrics: {e}")
    
    @contextmanager
    def track_request(self, endpoint: str, user_id: str = None):
        """
        Context manager for tracking API request metrics.
        
        Usage:
            with metrics_collector.track_request("/recommendations/123"):
                # Your API logic here
                pass
        """
        start_time = time.time()
        tags = {"endpoint": endpoint}
        if user_id:
            tags["user_id"] = user_id
        
        try:
            yield
            # Success case
            response_time = time.time() - start_time
            self.record_api_request(endpoint, response_time, success=True, tags=tags)
        except Exception as e:
            # Ошибка case
            response_time = time.time() - start_time
            self.record_api_request(endpoint, response_time, success=False, tags=tags)
            raise
    
    def record_api_request(self, endpoint: str, response_time: float, 
                          success: bool = True, tags: Dict[str, str] = None):
        """Record API request metrics."""
        with self._lock:
            # Global metrics
            self.api_metrics.request_count += 1
            self.api_metrics.total_response_time += response_time
            self.api_metrics.min_response_time = min(
                self.api_metrics.min_response_time, response_time
            )
            self.api_metrics.max_response_time = max(
                self.api_metrics.max_response_time, response_time
            )
            
            if not success:
                self.api_metrics.error_count += 1
            
            # Эндпоинт-specific metrics
            endpoint_metric = self.endpoint_metrics[endpoint]
            endpoint_metric.request_count += 1
            endpoint_metric.total_response_time += response_time
            endpoint_metric.min_response_time = min(
                endpoint_metric.min_response_time, response_time
            )
            endpoint_metric.max_response_time = max(
                endpoint_metric.max_response_time, response_time
            )
            
            if not success:
                endpoint_metric.error_count += 1
            
            # Сохранение individual Метрика point
            metric_tags = tags or {}
            metric_tags.update({
                "endpoint": endpoint,
                "success": str(success)
            })
            
            self._add_metric_point("api_response_time", response_time, metric_tags)
            self._add_metric_point("api_request_count", 1, metric_tags)
    
    def record_recommendation_shown(self, user_id: str, item_ids: List[str], 
                                  algorithm: str = "unknown"):
        """Record when recommendations are shown to user."""
        with self._lock:
            self.business_metrics.recommendations_shown += len(item_ids)
            self.business_metrics.unique_items_recommended.update(item_ids)
            
            tags = {
                "user_id": user_id,
                "algorithm": algorithm,
                "recommendation_count": str(len(item_ids))
            }
            
            self._add_metric_point("recommendations_shown", len(item_ids), tags)
    
    def record_recommendation_click(self, user_id: str, item_id: str, 
                                  position: int = None):
        """Record when user clicks on a recommendation."""
        with self._lock:
            self.business_metrics.recommendations_clicked += 1
            
            tags = {"user_id": user_id, "item_id": item_id}
            if position is not None:
                tags["position"] = str(position)
            
            self._add_metric_point("recommendation_click", 1, tags)
    
    def record_cart_addition(self, user_id: str, item_id: str, 
                           from_recommendation: bool = True):
        """Record when item is added to cart."""
        with self._lock:
            if from_recommendation:
                self.business_metrics.items_added_to_cart += 1
            
            tags = {
                "user_id": user_id,
                "item_id": item_id,
                "from_recommendation": str(from_recommendation)
            }
            
            self._add_metric_point("cart_addition", 1, tags)
    
    def record_model_prediction(self, model_name: str, prediction_time: float,
                              confidence_score: float = None):
        """Record model prediction metrics."""
        tags = {"model_name": model_name}
        
        self._add_metric_point("model_prediction_time", prediction_time, tags)
        
        if confidence_score is not None:
            self._add_metric_point("model_confidence", confidence_score, tags)
    
    def set_catalog_size(self, size: int):
        """Set total catalog size for coverage calculation."""
        with self._lock:
            self.business_metrics.total_catalog_size = size
    
    def _add_metric_point(self, name: str, value: float, tags: Dict[str, str] = None):
        """Add a metric point to history."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        self._metrics_history.append(point)
        self._current_window[name].append(point)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current aggregated metrics."""
        with self._lock:
            return {
                "api_metrics": {
                    "request_count": self.api_metrics.request_count,
                    "error_count": self.api_metrics.error_count,
                    "error_rate": self.api_metrics.error_rate,
                    "avg_response_time": self.api_metrics.avg_response_time,
                    "min_response_time": self.api_metrics.min_response_time,
                    "max_response_time": self.api_metrics.max_response_time,
                },
                "business_metrics": {
                    "recommendations_shown": self.business_metrics.recommendations_shown,
                    "recommendations_clicked": self.business_metrics.recommendations_clicked,
                    "items_added_to_cart": self.business_metrics.items_added_to_cart,
                    "ctr": self.business_metrics.ctr,
                    "conversion_rate": self.business_metrics.conversion_rate,
                    "catalog_coverage": self.business_metrics.catalog_coverage,
                    "unique_items_count": len(self.business_metrics.unique_items_recommended),
                },
                "endpoint_metrics": {
                    endpoint: {
                        "request_count": metrics.request_count,
                        "error_count": metrics.error_count,
                        "error_rate": metrics.error_rate,
                        "avg_response_time": metrics.avg_response_time,
                    }
                    for endpoint, metrics in self.endpoint_metrics.items()
                },
                "timestamp": datetime.now().isoformat()
            }
    
    def get_metrics_history(self, metric_name: str = None, 
                          since: datetime = None) -> List[MetricPoint]:
        """Get historical metrics data."""
        with self._lock:
            history = list(self._metrics_history)
            
            if metric_name:
                history = [p for p in history if p.name == metric_name]
            
            if since:
                history = [p for p in history if p.timestamp >= since]
            
            return history
    
    def get_metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for the specified time window."""
        since = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = self.get_metrics_history(since=since)
        
        summary = defaultdict(lambda: {"count": 0, "sum": 0, "min": float('inf'), "max": 0})
        
        for metric in recent_metrics:
            name = metric.name
            value = metric.value
            
            summary[name]["count"] += 1
            summary[name]["sum"] += value
            summary[name]["min"] = min(summary[name]["min"], value)
            summary[name]["max"] = max(summary[name]["max"], value)
        
        # Расчет averages
        result = {}
        for name, stats in summary.items():
            result[name] = {
                "count": stats["count"],
                "sum": stats["sum"],
                "avg": stats["sum"] / max(stats["count"], 1),
                "min": stats["min"] if stats["min"] != float('inf') else 0,
                "max": stats["max"]
            }
        
        return result
    
    def export_metrics(self, export_path: str = None):
        """Export metrics to file or external system."""
        if export_path is None:
            export_path = "logs/metrics"
        
        # Ensure directory exists
        Path(export_path).mkdir(parents=True, exist_ok=True)
        
        # Export current metrics
        current_metrics = self.get_current_metrics()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON export
        json_path = Path(export_path) / f"metrics_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(current_metrics, f, indent=2, default=str)
        
        # Prometheus Форматирование export
        prometheus_path = Path(export_path) / "metrics.prom"
        self._export_prometheus_format(prometheus_path, current_metrics)
        
        logger.info(f"Metrics exported to {export_path}")
    
    def _export_prometheus_format(self, file_path: Path, metrics: Dict[str, Any]):
        """Export metrics in Prometheus format."""
        lines = []
        
        # API metrics
        api_metrics = metrics["api_metrics"]
        lines.extend([
            f"# HELP api_requests_total Total number of API requests",
            f"# TYPE api_requests_total counter",
            f"api_requests_total {api_metrics['request_count']}",
            f"",
            f"# HELP api_errors_total Total number of API errors",
            f"# TYPE api_errors_total counter",
            f"api_errors_total {api_metrics['error_count']}",
            f"",
            f"# HELP api_response_time_seconds API response time in seconds",
            f"# TYPE api_response_time_seconds histogram",
            f"api_response_time_seconds_sum {api_metrics['avg_response_time']}",
            f"api_response_time_seconds_count {api_metrics['request_count']}",
            f"",
        ])
        
        # Business metrics
        business_metrics = metrics["business_metrics"]
        lines.extend([
            f"# HELP recommendations_shown_total Total recommendations shown",
            f"# TYPE recommendations_shown_total counter",
            f"recommendations_shown_total {business_metrics['recommendations_shown']}",
            f"",
            f"# HELP recommendations_clicked_total Total recommendation clicks",
            f"# TYPE recommendations_clicked_total counter",
            f"recommendations_clicked_total {business_metrics['recommendations_clicked']}",
            f"",
            f"# HELP recommendation_ctr Click-through rate of recommendations",
            f"# TYPE recommendation_ctr gauge",
            f"recommendation_ctr {business_metrics['ctr']:.4f}",
            f"",
            f"# HELP catalog_coverage Percentage of catalog covered by recommendations",
            f"# TYPE catalog_coverage gauge",
            f"catalog_coverage {business_metrics['catalog_coverage']:.4f}",
            f"",
        ])
        
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self.api_metrics = APIMetrics()
            self.business_metrics = BusinessMetrics()
            self.endpoint_metrics.clear()
            self._metrics_history.clear()
            self._current_window.clear()
    
    def __del__(self):
        """Cleanup when collector is destroyed."""
        self.stop_export_thread()


# Global metrics collector instance
metrics_collector = MetricsCollector()


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_thresholds = {
            "max_error_rate": 0.05,  # 5% error rate threshold
            "max_response_time": 1.0,  # 1 second response time threshold
            "min_ctr": 0.01,  # 1% minimum CTR
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        current_metrics = self.metrics_collector.get_current_metrics()
        
        health_status = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # API health checks
        api_metrics = current_metrics["api_metrics"]
        
        # Ошибка rate Проверка
        error_rate_healthy = api_metrics["error_rate"] <= self.health_thresholds["max_error_rate"]
        health_status["checks"]["error_rate"] = {
            "healthy": error_rate_healthy,
            "value": api_metrics["error_rate"],
            "threshold": self.health_thresholds["max_error_rate"]
        }
        
        # Ответ time Проверка
        response_time_healthy = api_metrics["avg_response_time"] <= self.health_thresholds["max_response_time"]
        health_status["checks"]["response_time"] = {
            "healthy": response_time_healthy,
            "value": api_metrics["avg_response_time"],
            "threshold": self.health_thresholds["max_response_time"]
        }
        
        # Business metrics Проверка
        business_metrics = current_metrics["business_metrics"]
        ctr_healthy = (business_metrics["ctr"] >= self.health_thresholds["min_ctr"] 
                      if business_metrics["recommendations_shown"] > 100 else True)
        
        health_status["checks"]["ctr"] = {
            "healthy": ctr_healthy,
            "value": business_metrics["ctr"],
            "threshold": self.health_thresholds["min_ctr"]
        }
        
        # Overall health Статус
        all_healthy = all(check["healthy"] for check in health_status["checks"].values())
        health_status["status"] = "healthy" if all_healthy else "unhealthy"
        
        return health_status


# Global health checker
health_checker = HealthChecker(metrics_collector)
