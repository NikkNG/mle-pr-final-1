"""
API Endpoints for Automatic Retraining Management

Provides REST API endpoints for monitoring and controlling automatic retraining.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from ..retraining.retraining_service import RetrainingService

logger = logging.getLogger(__name__)

# Global retraining Сервис instance
retraining_service: Optional[RetrainingService] = None

def get_retraining_service() -> RetrainingService:
    """Get retraining service instance."""
    global retraining_service
    if retraining_service is None:
        raise HTTPException(status_code=503, detail="Retraining service not initialized")
    return retraining_service

def initialize_retraining_service(service: RetrainingService):
    """Initialize global retraining service."""
    global retraining_service
    retraining_service = service

# Pydantic models for API
class MetricRecord(BaseModel):
    """Model for recording metrics."""
    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Metric value")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class ManualRetrainingRequest(BaseModel):
    """Model for manual retraining request."""
    reason: str = Field(..., description="Reason for retraining")
    models: Optional[List[str]] = Field(default=None, description="Models to retrain")
    auto_deploy: bool = Field(default=False, description="Auto-deploy if successful")

class DeploymentRequest(BaseModel):
    """Model for deployment request."""
    model_name: str = Field(..., description="Model name to deploy")
    model_path: str = Field(..., description="Path to model file")
    strategy: str = Field(default="blue_green", description="Deployment strategy")

class ScheduleRequest(BaseModel):
    """Model for adding/updating schedules."""
    name: str = Field(..., description="Schedule name")
    schedule_type: str = Field(..., description="Schedule type (daily, weekly, hourly)")
    schedule_value: str = Field(..., description="Schedule value")
    models: Optional[List[str]] = Field(default=None, description="Models to retrain")
    enabled: bool = Field(default=True, description="Whether schedule is enabled")

class BaselineRequest(BaseModel):
    """Model for setting metric baselines."""
    metric_name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Baseline value")

# Создание router
router = APIRouter(prefix="/retraining", tags=["retraining"])

@router.get("/status")
async def get_retraining_status(
    service: RetrainingService = Depends(get_retraining_service)
) -> Dict[str, Any]:
    """Get comprehensive retraining service status."""
    try:
        return service.get_service_status()
    except Exception as e:
        logger.error(f"Error getting retraining status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics/record")
async def record_metric(
    metric: MetricRecord,
    service: RetrainingService = Depends(get_retraining_service)
):
    """Record a performance metric."""
    try:
        service.record_metric(
            metric_name=metric.metric_name,
            value=metric.value,
            metadata=metric.metadata
        )
        return {"status": "success", "message": "Metric recorded"}
    except Exception as e:
        logger.error(f"Error recording metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics/baseline")
async def set_baseline(
    baseline: BaselineRequest,
    service: RetrainingService = Depends(get_retraining_service)
):
    """Set performance baseline for a metric."""
    try:
        service.set_baseline(baseline.metric_name, baseline.value)
        return {"status": "success", "message": "Baseline set"}
    except Exception as e:
        logger.error(f"Error setting baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/summary")
async def get_metrics_summary(
    hours: int = 24,
    service: RetrainingService = Depends(get_retraining_service)
) -> Dict[str, Any]:
    """Get metrics summary for specified time period."""
    try:
        return service.get_metrics_summary(hours)
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trigger")
async def trigger_manual_retraining(
    request: ManualRetrainingRequest,
    background_tasks: BackgroundTasks,
    service: RetrainingService = Depends(get_retraining_service)
) -> Dict[str, Any]:
    """Manually trigger model retraining."""
    try:
        # Run retraining in background
        result = service.trigger_manual_retraining(
            reason=request.reason,
            models=request.models,
            auto_deploy=request.auto_deploy
        )
        
        return {
            "status": "triggered",
            "message": "Retraining triggered successfully",
            "trigger_id": result.get('deployment_id', 'unknown'),
            "details": result
        }
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/training")
async def get_training_history(
    limit: int = 10,
    service: RetrainingService = Depends(get_retraining_service)
) -> List[Dict[str, Any]]:
    """Get retraining history."""
    try:
        return service.get_retraining_history(limit)
    except Exception as e:
        logger.error(f"Error getting training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/deployment")
async def get_deployment_history(
    model_name: Optional[str] = None,
    limit: int = 10,
    service: RetrainingService = Depends(get_retraining_service)
) -> List[Dict[str, Any]]:
    """Get deployment history."""
    try:
        return service.get_deployment_history(model_name, limit)
    except Exception as e:
        logger.error(f"Error getting deployment history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deploy")
async def deploy_model(
    request: DeploymentRequest,
    service: RetrainingService = Depends(get_retraining_service)
) -> Dict[str, Any]:
    """Deploy a trained model."""
    try:
        result = service.deploy_model(
            model_name=request.model_name,
            model_path=request.model_path,
            strategy=request.strategy
        )
        return result
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollback/{model_name}")
async def rollback_model(
    model_name: str,
    service: RetrainingService = Depends(get_retraining_service)
) -> Dict[str, Any]:
    """Rollback model to previous version."""
    try:
        result = service.rollback_model(model_name)
        return result
    except Exception as e:
        logger.error(f"Error rolling back model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/schedules")
async def get_schedules(
    service: RetrainingService = Depends(get_retraining_service)
) -> Dict[str, Any]:
    """Get all retraining schedules."""
    try:
        return service.scheduler.get_schedules()
    except Exception as e:
        logger.error(f"Error getting schedules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/schedules")
async def add_schedule(
    request: ScheduleRequest,
    service: RetrainingService = Depends(get_retraining_service)
):
    """Add a new retraining schedule."""
    try:
        service.scheduler.add_schedule(
            name=request.name,
            schedule_type=request.schedule_type,
            schedule_value=request.schedule_value,
            models=request.models,
            enabled=request.enabled
        )
        return {"status": "success", "message": "Schedule added"}
    except Exception as e:
        logger.error(f"Error adding schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/schedules/{schedule_name}")
async def remove_schedule(
    schedule_name: str,
    service: RetrainingService = Depends(get_retraining_service)
):
    """Remove a retraining schedule."""
    try:
        service.scheduler.remove_schedule(schedule_name)
        return {"status": "success", "message": "Schedule removed"}
    except Exception as e:
        logger.error(f"Error removing schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/schedules/{schedule_name}/enable")
async def enable_schedule(
    schedule_name: str,
    service: RetrainingService = Depends(get_retraining_service)
):
    """Enable a retraining schedule."""
    try:
        service.scheduler.enable_schedule(schedule_name)
        return {"status": "success", "message": "Schedule enabled"}
    except Exception as e:
        logger.error(f"Error enabling schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/schedules/{schedule_name}/disable")
async def disable_schedule(
    schedule_name: str,
    service: RetrainingService = Depends(get_retraining_service)
):
    """Disable a retraining schedule."""
    try:
        service.scheduler.disable_schedule(schedule_name)
        return {"status": "success", "message": "Schedule disabled"}
    except Exception as e:
        logger.error(f"Error disabling schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/triggers/history")
async def get_trigger_history(
    hours: int = 24,
    service: RetrainingService = Depends(get_retraining_service)
) -> List[Dict[str, Any]]:
    """Get trigger history."""
    try:
        return service.scheduler.get_trigger_history(hours)
    except Exception as e:
        logger.error(f"Error getting trigger history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/report")
async def get_performance_report(
    days: int = 7,
    service: RetrainingService = Depends(get_retraining_service)
) -> Dict[str, Any]:
    """Get performance report for retraining system."""
    try:
        return service.scheduler.get_performance_report(days)
    except Exception as e:
        logger.error(f"Error getting performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start")
async def start_retraining_service(
    service: RetrainingService = Depends(get_retraining_service)
):
    """Start the retraining service."""
    try:
        service.start()
        return {"status": "success", "message": "Retraining service started"}
    except Exception as e:
        logger.error(f"Error starting retraining service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_retraining_service(
    service: RetrainingService = Depends(get_retraining_service)
):
    """Stop the retraining service."""
    try:
        service.stop()
        return {"status": "success", "message": "Retraining service stopped"}
    except Exception as e:
        logger.error(f"Error stopping retraining service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_retraining_health(
    service: RetrainingService = Depends(get_retraining_service)
) -> Dict[str, Any]:
    """Get health status of retraining system."""
    try:
        status = service.get_service_status()
        
        # Determine overall health
        overall_health = "healthy"
        components = status.get('components', {})
        
        # Проверка metrics Мониторинг health
        metrics_health = components.get('metrics_monitor', {}).get('health_status', {})
        if metrics_health.get('overall_status') != 'healthy':
            overall_health = "degraded"
        
        # Проверка if any Компонент is not running
        if not status.get('service_running', False):
            overall_health = "unhealthy"
        
        return {
            "status": overall_health,
            "timestamp": datetime.now().isoformat(),
            "service_running": status.get('service_running', False),
            "components_status": {
                "metrics_monitor": components.get('metrics_monitor', {}).get('running', False),
                "scheduler": components.get('scheduler', {}).get('is_running', False),
                "model_trainer": not components.get('model_trainer', {}).get('is_training', False),
                "deployment_manager": not components.get('deployment_manager', {}).get('is_deploying', False)
            },
            "recent_activity": status.get('recent_activity', {}),
            "details": status
        }
    except Exception as e:
        logger.error(f"Error getting retraining health: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        } 