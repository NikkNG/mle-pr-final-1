"""
Automatic Retraining Service

Main service that orchestrates automatic model retraining workflow.
"""

import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .metrics_monitor import MetricsMonitor, create_default_metrics_config
from .model_trainer import ModelTrainer
from .scheduler import RetrainingScheduler
from .deployment_manager import DeploymentManager

logger = logging.getLogger(__name__)

class RetrainingService:
    """
    Main service for automatic model retraining.
    
    Orchestrates:
    - Metrics monitoring
    - Scheduled retraining
    - Model training and validation
    - Safe deployment with rollback
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        data_path: str = "data/",
        models_path: str = "models/",
        logs_path: str = "logs/"
    ):
        self.config = config or self._get_default_config()
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.logs_path = Path(logs_path)
        
        # Создание directories
        for path in [self.data_path, self.models_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Инициализация components
        self._initialize_components()
        
        # Сервис Состояние
        self.is_running = False
        self.service_thread: Optional[threading.Thread] = None
        
        logger.info("RetrainingService initialized")
    
    def _initialize_components(self):
        """Initialize all retraining components."""
        try:
            # Metrics Мониторинг
            metrics_config = create_default_metrics_config()
            self.metrics_monitor = MetricsMonitor(
                metrics_config=metrics_config,
                storage_path=str(self.logs_path / "metrics"),
                check_interval=self.config.get('metrics_check_interval', 300)
            )
            
            # Модель Trainer
            self.model_trainer = ModelTrainer(
                data_path=str(self.data_path),
                models_path=str(self.models_path),
                experiments_path=str(self.logs_path / "experiments"),
                mlflow_tracking_uri=self.config.get('mlflow_uri', 'http://localhost:5000')
            )
            
            # Развертывание Manager
            self.deployment_manager = DeploymentManager(
                models_path=str(self.models_path),
                deployment_path=str(self.models_path / "production"),
                backup_path=str(self.models_path / "backup")
            )
            
            # Retraining Scheduler
            self.scheduler = RetrainingScheduler(
                config_path=str(self.logs_path / "config" / "retraining_config.json"),
                logs_path=str(self.logs_path / "retraining")
            )
            
            # Wire components together
            self._wire_components()
            
            logger.info("All retraining components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _wire_components(self):
        """Wire components together with callbacks."""
        # Set retraining Колбэк for scheduler
        self.scheduler.set_retraining_callback(self._handle_retraining_trigger)
        
        # Set metrics Мониторинг and Модель trainer for scheduler
        self.scheduler.set_metrics_monitor(self.metrics_monitor)
        self.scheduler.set_model_trainer(self.model_trainer)
        
        # Set health Проверка Колбэк for Развертывание manager
        self.deployment_manager.set_health_check_callback(self._deployment_health_check)
    
    def start(self):
        """Start the retraining service."""
        if self.is_running:
            logger.warning("Retraining service is already running")
            return
        
        try:
            # Запуск metrics Мониторинг
            self.metrics_monitor.start_monitoring()
            
            # Запуск scheduler
            self.scheduler.start()
            
            # Запуск Сервис Мониторинг thread
            self.is_running = True
            self.service_thread = threading.Thread(target=self._service_loop, daemon=True)
            self.service_thread.start()
            
            logger.info("Retraining service started successfully")
            
        except Exception as e:
            logger.error(f"Error starting retraining service: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the retraining service."""
        logger.info("Stopping retraining service...")
        
        self.is_running = False
        
        # Остановка components
        if hasattr(self, 'metrics_monitor'):
            self.metrics_monitor.stop_monitoring()
        
        if hasattr(self, 'scheduler'):
            self.scheduler.stop()
        
        # Wait for Сервис thread
        if self.service_thread:
            self.service_thread.join(timeout=10)
        
        logger.info("Retraining service stopped")
    
    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        self.metrics_monitor.record_metric(metric_name, value, metadata)
    
    def set_baseline(self, metric_name: str, value: float):
        """Set performance baseline for a metric."""
        self.metrics_monitor.set_baseline(metric_name, value)
    
    def trigger_manual_retraining(
        self,
        reason: str = "Manual trigger",
        models: List[str] = None,
        auto_deploy: bool = False
    ) -> Dict[str, Any]:
        """Manually trigger model retraining."""
        return self.scheduler.trigger_manual_retraining(reason, models)
    
    def deploy_model(
        self,
        model_name: str,
        model_path: str,
        strategy: str = "blue_green"
    ) -> Dict[str, Any]:
        """Deploy a trained model."""
        return self.deployment_manager.deploy_model(model_name, model_path, strategy)
    
    def rollback_model(self, model_name: str) -> Dict[str, Any]:
        """Rollback model to previous version."""
        return self.deployment_manager.rollback_to_previous(model_name)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        try:
            return {
                'service_running': self.is_running,
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'metrics_monitor': {
                        'running': self.metrics_monitor.is_monitoring,
                        'metrics_count': len(self.metrics_monitor.metrics_history),
                        'health_status': self.metrics_monitor.check_model_health()
                    },
                    'scheduler': self.scheduler.get_status(),
                    'model_trainer': self.model_trainer.get_training_status(),
                    'deployment_manager': self.deployment_manager.get_deployment_status()
                },
                'recent_activity': self._get_recent_activity()
            }
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {'error': str(e)}
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary."""
        return self.metrics_monitor.get_all_metrics_summary(hours)
    
    def get_retraining_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get retraining history."""
        return self.model_trainer.training_history[-limit:]
    
    def get_deployment_history(self, model_name: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return self.deployment_manager.get_deployment_history(model_name, limit)
    
    def _handle_retraining_trigger(self, trigger_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle retraining trigger from scheduler or metrics monitor."""
        try:
            logger.info(f"Handling retraining trigger: {trigger_info['reason']}")
            
            # Determine models to retrain
            models_to_retrain = trigger_info.get('models')
            if not models_to_retrain:
                # Default models based on trigger type
                if trigger_info.get('trigger_type') == 'scheduled':
                    models_to_retrain = ['popularity', 'als', 'hybrid']
                else:
                    models_to_retrain = ['hybrid']  # Quick retrain for metric issues
            
            # Trigger retraining
            training_result = self.model_trainer.trigger_retraining(
                trigger_info=trigger_info,
                models_to_retrain=models_to_retrain
            )
            
            # Обработка Развертывание based on results
            if training_result['status'] == 'completed':
                deployment_results = self._handle_post_training_deployment(
                    training_result, trigger_info
                )
                training_result['deployment_results'] = deployment_results
            
            return training_result
            
        except Exception as e:
            logger.error(f"Error handling retraining trigger: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _handle_post_training_deployment(
        self,
        training_result: Dict[str, Any],
        trigger_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle deployment after successful training."""
        deployment_results = {}
        
        try:
            deployment_decision = training_result.get('deployment_decision', {})
            
            if deployment_decision.get('deploy', False):
                recommended_model = deployment_decision['recommended_model']
                
                # Get Модель path from training results
                model_info = training_result['training_results'].get(recommended_model, {})
                model_path = model_info.get('model_path')
                
                if model_path:
                    # Determine Развертывание Стратегия
                    strategy = "blue_green"
                    if deployment_decision.get('ab_test_recommended', False):
                        strategy = "canary"
                    
                    # Deploy Модель
                    deployment_result = self.deployment_manager.deploy_model(
                        model_name=recommended_model,
                        model_path=model_path,
                        deployment_strategy=strategy,
                        auto_rollback=True
                    )
                    
                    deployment_results[recommended_model] = deployment_result
                    
                    logger.info(f"Deployed model {recommended_model}: {deployment_result['status']}")
                else:
                    deployment_results[recommended_model] = {
                        'status': 'failed',
                        'reason': 'Model path not found'
                    }
            else:
                logger.info(f"Deployment not recommended: {deployment_decision.get('reason')}")
                deployment_results['decision'] = 'no_deployment'
                deployment_results['reason'] = deployment_decision.get('reason')
        
        except Exception as e:
            logger.error(f"Error in post-training deployment: {e}")
            deployment_results['error'] = str(e)
        
        return deployment_results
    
    def _deployment_health_check(self, model_name: str) -> Dict[str, Any]:
        """Health check callback for deployment manager."""
        try:
            # Get recent metrics for the Модель
            metrics_summary = self.metrics_monitor.get_all_metrics_summary(hours=1)
            
            # Проверка if metrics are within acceptable ranges
            health_issues = []
            
            for metric_name, summary in metrics_summary.items():
                if summary and 'latest' in summary:
                    latest_value = summary['latest']
                    
                    # Проверка against thresholds
                    if metric_name == 'error_rate' and latest_value > 0.1:
                        health_issues.append(f"High error rate: {latest_value}")
                    elif metric_name == 'response_time' and latest_value > 1.0:
                        health_issues.append(f"High response time: {latest_value}")
                    elif metric_name in ['precision_at_10', 'recall_at_10', 'ndcg_at_10'] and latest_value < 0.05:
                        health_issues.append(f"Low {metric_name}: {latest_value}")
            
            if health_issues:
                return {
                    'healthy': False,
                    'reason': '; '.join(health_issues),
                    'metrics_checked': list(metrics_summary.keys())
                }
            else:
                return {
                    'healthy': True,
                    'metrics_checked': list(metrics_summary.keys())
                }
                
        except Exception as e:
            logger.error(f"Error in deployment health check: {e}")
            return {'healthy': False, 'reason': f'Health check error: {str(e)}'}
    
    def _service_loop(self):
        """Main service monitoring loop."""
        while self.is_running:
            try:
                # Periodic maintenance tasks
                self._perform_maintenance()
                
                # Sleep for Мониторинг interval
                time.sleep(self.config.get('service_check_interval', 600))  # 10 minutes
                
            except Exception as e:
                logger.error(f"Error in service loop: {e}")
                time.sleep(60)  # Short sleep on error
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks."""
        try:
            # Очистка up old Развертывание backups
            self.deployment_manager.cleanup_old_backups()
            
            # Логирование Сервис health
            status = self.get_service_status()
            logger.info(f"Service health check: {status['components']['metrics_monitor']['health_status']['overall_status']}")
            
        except Exception as e:
            logger.error(f"Error in maintenance: {e}")
    
    def _get_recent_activity(self) -> Dict[str, Any]:
        """Get recent activity summary."""
        try:
            return {
                'recent_triggers': self.scheduler.get_trigger_history(hours=24),
                'recent_trainings': self.get_retraining_history(limit=5),
                'recent_deployments': self.get_deployment_history(limit=5)
            }
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default service configuration."""
        return {
            'metrics_check_interval': 300,  # 5 minutes
            'service_check_interval': 600,  # 10 minutes
            'mlflow_uri': 'http://localhost:5000',
            'auto_deploy_threshold': 0.05,  # 5% improvement
            'max_concurrent_trainings': 1,
            'deployment_strategy': 'blue_green'
        }


def create_retraining_service(config_path: str = None) -> RetrainingService:
    """Create and configure retraining service."""
    config = {}
    
    if config_path:
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
    
    return RetrainingService(config=config) 