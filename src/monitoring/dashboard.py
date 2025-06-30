"""
Monitoring dashboard and API endpoints for the recommendation system.

Provides REST API endpoints for accessing metrics data and health status.
Can be integrated with external monitoring systems like Grafana or Prometheus.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from .metrics import metrics_collector, health_checker

logger = logging.getLogger(__name__)

# Создание router for Мониторинг endpoints
monitoring_router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# Pydantic models for API responses
class HealthCheckResponse(BaseModel):
    status: str
    checks: Dict[str, Any]
    timestamp: str


class MetricsResponse(BaseModel):
    api_metrics: Dict[str, Any]
    business_metrics: Dict[str, Any]
    endpoint_metrics: Dict[str, Any]
    timestamp: str


class MetricsSummaryResponse(BaseModel):
    window_minutes: int
    metrics: Dict[str, Any]
    timestamp: str


class AlertRequest(BaseModel):
    metric_name: str
    threshold: float
    operator: str = Field(..., pattern="^(gt|lt|gte|lte|eq)$")
    message: str = ""


# API Endpoints
@monitoring_router.get("/health", response_model=HealthCheckResponse)
async def get_health_status():
    """
    Get comprehensive health status of the recommendation system.
    
    Returns:
        Health status with individual component checks
    """
    try:
        health_status = health_checker.check_health()
        return HealthCheckResponse(**health_status)
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@monitoring_router.get("/metrics", response_model=MetricsResponse)
async def get_current_metrics():
    """
    Get current aggregated metrics for all components.
    
    Returns:
        Current API metrics, business metrics, and endpoint-specific metrics
    """
    try:
        current_metrics = metrics_collector.get_current_metrics()
        return MetricsResponse(**current_metrics)
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@monitoring_router.get("/metrics/summary", response_model=MetricsSummaryResponse)
async def get_metrics_summary(
    window_minutes: int = Query(60, ge=1, le=1440, description="Time window in minutes")
):
    """
    Get metrics summary for a specified time window.
    
    Args:
        window_minutes: Time window in minutes (1-1440)
    
    Returns:
        Aggregated metrics summary for the time window
    """
    try:
        summary = metrics_collector.get_metrics_summary(window_minutes)
        return MetricsSummaryResponse(
            window_minutes=window_minutes,
            metrics=summary,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics summary")


@monitoring_router.get("/metrics/history/{metric_name}")
async def get_metric_history(
    metric_name: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
    format: str = Query("json", pattern="^(json|csv)$", description="Response format")
):
    """
    Get historical data for a specific metric.
    
    Args:
        metric_name: Name of the metric to retrieve
        hours: Number of hours of history (1-168)
        format: Response format (json or csv)
    
    Returns:
        Historical metric data
    """
    try:
        since = datetime.now() - timedelta(hours=hours)
        history = metrics_collector.get_metrics_history(metric_name, since)
        
        if format == "csv":
            # Преобразование to CSV Форматирование
            csv_lines = ["timestamp,value,tags"]
            for point in history:
                tags_str = ";".join([f"{k}={v}" for k, v in point.tags.items()])
                csv_lines.append(f"{point.timestamp.isoformat()},{point.value},{tags_str}")
            
            return {"data": "\n".join(csv_lines), "format": "csv"}
        
        # JSON Форматирование
        return {
            "metric_name": metric_name,
            "hours": hours,
            "data": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "tags": point.tags
                }
                for point in history
            ],
            "count": len(history)
        }
    
    except Exception as e:
        logger.error(f"Error getting metric history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metric history")


@monitoring_router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus format for scraping.
    
    Returns:
        Metrics in Prometheus exposition format
    """
    try:
        # Export metrics temporarily to get Prometheus Форматирование
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_collector.export_metrics(temp_dir)
            prometheus_file = os.path.join(temp_dir, "metrics.prom")
            
            if os.path.exists(prometheus_file):
                with open(prometheus_file, 'r') as f:
                    content = f.read()
                return {"metrics": content, "content_type": "text/plain"}
            else:
                raise HTTPException(status_code=500, detail="Failed to generate Prometheus metrics")
    
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate Prometheus metrics")


@monitoring_router.post("/metrics/record/recommendation")
async def record_recommendation_event(
    user_id: str,
    item_ids: List[str],
    algorithm: str = "unknown",
    background_tasks: BackgroundTasks = None
):
    """
    Record a recommendation event.
    
    Args:
        user_id: ID of the user
        item_ids: List of recommended item IDs
        algorithm: Algorithm used for recommendation
    """
    try:
        metrics_collector.record_recommendation_shown(user_id, item_ids, algorithm)
        return {"status": "recorded", "user_id": user_id, "item_count": len(item_ids)}
    except Exception as e:
        logger.error(f"Error recording recommendation event: {e}")
        raise HTTPException(status_code=500, detail="Failed to record recommendation event")


@monitoring_router.post("/metrics/record/click")
async def record_click_event(
    user_id: str,
    item_id: str,
    position: Optional[int] = None
):
    """
    Record a recommendation click event.
    
    Args:
        user_id: ID of the user
        item_id: ID of the clicked item
        position: Position of the item in recommendations
    """
    try:
        metrics_collector.record_recommendation_click(user_id, item_id, position)
        return {"status": "recorded", "user_id": user_id, "item_id": item_id}
    except Exception as e:
        logger.error(f"Error recording click event: {e}")
        raise HTTPException(status_code=500, detail="Failed to record click event")


@monitoring_router.post("/metrics/record/cart")
async def record_cart_event(
    user_id: str,
    item_id: str,
    from_recommendation: bool = True
):
    """
    Record a cart addition event.
    
    Args:
        user_id: ID of the user
        item_id: ID of the item added to cart
        from_recommendation: Whether the item was from a recommendation
    """
    try:
        metrics_collector.record_cart_addition(user_id, item_id, from_recommendation)
        return {"status": "recorded", "user_id": user_id, "item_id": item_id}
    except Exception as e:
        logger.error(f"Error recording cart event: {e}")
        raise HTTPException(status_code=500, detail="Failed to record cart event")


@monitoring_router.post("/metrics/reset")
async def reset_metrics():
    """
    Reset all metrics (for testing purposes).
    
    WARNING: This will clear all accumulated metrics data.
    """
    try:
        metrics_collector.reset_metrics()
        return {"status": "reset", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset metrics")


@monitoring_router.get("/dashboard")
async def get_dashboard_data():
    """
    Get comprehensive dashboard data for monitoring UI.
    
    Returns:
        Complete dashboard data including metrics, health, and trends
    """
    try:
        # Get current metrics
        current_metrics = metrics_collector.get_current_metrics()
        
        # Get health Статус
        health_status = health_checker.check_health()
        
        # Get recent trends (last hour)
        trends = metrics_collector.get_metrics_summary(60)
        
        # Get top endpoints by Запрос count
        endpoint_metrics = current_metrics.get("endpoint_metrics", {})
        top_endpoints = sorted(
            endpoint_metrics.items(),
            key=lambda x: x[1]["request_count"],
            reverse=True
        )[:10]
        
        dashboard_data = {
            "overview": {
                "status": health_status["status"],
                "total_requests": current_metrics["api_metrics"]["request_count"],
                "error_rate": current_metrics["api_metrics"]["error_rate"],
                "avg_response_time": current_metrics["api_metrics"]["avg_response_time"],
                "recommendations_shown": current_metrics["business_metrics"]["recommendations_shown"],
                "ctr": current_metrics["business_metrics"]["ctr"],
                "conversion_rate": current_metrics["business_metrics"]["conversion_rate"],
                "catalog_coverage": current_metrics["business_metrics"]["catalog_coverage"],
            },
            "health_checks": health_status["checks"],
            "trends": trends,
            "top_endpoints": dict(top_endpoints),
            "timestamp": datetime.now().isoformat()
        }
        
        return dashboard_data
    
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")


@monitoring_router.get("/alerts")
async def get_active_alerts():
    """
    Get currently active alerts based on metric thresholds.
    
    Returns:
        List of active alerts
    """
    try:
        current_metrics = metrics_collector.get_current_metrics()
        health_status = health_checker.check_health()
        
        alerts = []
        
        # Проверка for unhealthy components
        for check_name, check_data in health_status["checks"].items():
            if not check_data["healthy"]:
                alerts.append({
                    "type": "health_check",
                    "severity": "critical",
                    "message": f"{check_name} is unhealthy: {check_data['value']} > {check_data['threshold']}",
                    "timestamp": datetime.now().isoformat(),
                    "metric": check_name,
                    "value": check_data["value"],
                    "threshold": check_data["threshold"]
                })
        
        # Проверка for low business metrics
        business_metrics = current_metrics["business_metrics"]
        
        if business_metrics["recommendations_shown"] > 100:  # Only if we have enough data
            if business_metrics["ctr"] < 0.005:  # Less than 0.5% CTR
                alerts.append({
                    "type": "business_metric",
                    "severity": "warning",
                    "message": f"Low CTR detected: {business_metrics['ctr']:.4f}",
                    "timestamp": datetime.now().isoformat(),
                    "metric": "ctr",
                    "value": business_metrics["ctr"],
                    "threshold": 0.005
                })
            
            if business_metrics["catalog_coverage"] < 0.1:  # Less than 10% coverage
                alerts.append({
                    "type": "business_metric",
                    "severity": "warning",
                    "message": f"Low catalog coverage: {business_metrics['catalog_coverage']:.4f}",
                    "timestamp": datetime.now().isoformat(),
                    "metric": "catalog_coverage",
                    "value": business_metrics["catalog_coverage"],
                    "threshold": 0.1
                })
        
        return {"alerts": alerts, "count": len(alerts)}
    
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@monitoring_router.post("/export")
async def export_metrics(
    format: str = Query("json", pattern="^(json|prometheus|csv)$"),
    background_tasks: BackgroundTasks = None
):
    """
    Export current metrics in specified format.
    
    Args:
        format: Export format (json, prometheus, or csv)
    
    Returns:
        Export status and download information
    """
    try:
        if background_tasks:
            background_tasks.add_task(metrics_collector.export_metrics)
        else:
            metrics_collector.export_metrics()
        
        return {
            "status": "exported",
            "format": format,
            "timestamp": datetime.now().isoformat(),
            "export_path": "logs/metrics"
        }
    
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics") 