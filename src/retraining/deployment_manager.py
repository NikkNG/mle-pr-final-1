"""
Deployment Manager for Model Updates

Handles safe deployment of retrained models with rollback capabilities.
"""

import logging
import os
import shutil
import pickle
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

class DeploymentManager:
    """
    Manages safe deployment of retrained models.
    
    Features:
    - Blue-green deployment strategy
    - Automatic rollback on failure
    - Model version management
    - Health checks before deployment
    - Deployment history and auditing
    """
    
    def __init__(
        self,
        models_path: str = "models/",
        deployment_path: str = "models/production/",
        backup_path: str = "models/backup/",
        config_path: str = "config/deployment_config.json"
    ):
        self.models_path = Path(models_path)
        self.deployment_path = Path(deployment_path)
        self.backup_path = Path(backup_path)
        self.config_path = Path(config_path)
        
        # Создание directories
        for path in [self.models_path, self.deployment_path, self.backup_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Загрузка Конфигурация
        self.config = self._load_config()
        
        # Развертывание Состояние
        self.deployment_history: List[Dict[str, Any]] = []
        self.current_deployment: Optional[Dict[str, Any]] = None
        self.is_deploying = False
        
        # Health Проверка Колбэк
        self.health_check_callback: Optional[callable] = None
        
        logger.info("DeploymentManager initialized")
    
    def set_health_check_callback(self, callback: callable):
        """Set callback for health checks."""
        self.health_check_callback = callback
    
    def deploy_model(
        self,
        model_name: str,
        model_path: str,
        deployment_strategy: str = "blue_green",
        auto_rollback: bool = True,
        health_check_timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Deploy a new model version.
        
        Args:
            model_name: Name of the model to deploy
            model_path: Path to the new model file
            deployment_strategy: Deployment strategy ('blue_green', 'canary', 'immediate')
            auto_rollback: Whether to auto-rollback on failure
            health_check_timeout: Timeout for health checks in seconds
            
        Returns:
            Deployment result
        """
        if self.is_deploying:
            return {'status': 'failed', 'reason': 'Deployment already in progress'}
        
        deployment_id = f"deploy_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        deployment_info = {
            'deployment_id': deployment_id,
            'model_name': model_name,
            'model_path': model_path,
            'strategy': deployment_strategy,
            'auto_rollback': auto_rollback,
            'start_time': datetime.now(),
            'status': 'in_progress'
        }
        
        self.is_deploying = True
        self.current_deployment = deployment_info
        
        try:
            logger.info(f"Starting deployment: {deployment_id}")
            
            # Step 1: Валидация new Модель
            validation_result = self._validate_model(model_path, model_name)
            if not validation_result['valid']:
                raise Exception(f"Model validation failed: {validation_result['reason']}")
            
            deployment_info['validation_result'] = validation_result
            
            # Step 2: Резервное копирование current Модель
            backup_result = self._backup_current_model(model_name)
            deployment_info['backup_result'] = backup_result
            
            # Step 3: Deploy based on Стратегия
            if deployment_strategy == "blue_green":
                deploy_result = self._blue_green_deployment(model_name, model_path)
            elif deployment_strategy == "canary":
                deploy_result = self._canary_deployment(model_name, model_path)
            elif deployment_strategy == "immediate":
                deploy_result = self._immediate_deployment(model_name, model_path)
            else:
                raise Exception(f"Unknown deployment strategy: {deployment_strategy}")
            
            deployment_info['deploy_result'] = deploy_result
            
            # Step 4: Health Проверка
            health_result = self._perform_health_check(model_name, health_check_timeout)
            deployment_info['health_result'] = health_result
            
            if not health_result['healthy']:
                if auto_rollback:
                    logger.warning("Health check failed, initiating rollback")
                    rollback_result = self._rollback_deployment(model_name, backup_result)
                    deployment_info['rollback_result'] = rollback_result
                    deployment_info['status'] = 'rolled_back'
                else:
                    deployment_info['status'] = 'failed'
                    raise Exception(f"Health check failed: {health_result['reason']}")
            else:
                deployment_info['status'] = 'completed'
                logger.info(f"Deployment completed successfully: {deployment_id}")
            
            deployment_info['end_time'] = datetime.now()
            deployment_info['duration'] = (deployment_info['end_time'] - deployment_info['start_time']).total_seconds()
            
            # Обновление Развертывание history
            self.deployment_history.append(deployment_info)
            self._save_deployment_history()
            
            return {
                'status': deployment_info['status'],
                'deployment_id': deployment_id,
                'details': deployment_info
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            
            deployment_info['status'] = 'failed'
            deployment_info['error'] = str(e)
            deployment_info['end_time'] = datetime.now()
            
            # Attempt rollback if auto_rollback Включен
            if auto_rollback and 'backup_result' in deployment_info:
                try:
                    rollback_result = self._rollback_deployment(model_name, deployment_info['backup_result'])
                    deployment_info['rollback_result'] = rollback_result
                    deployment_info['status'] = 'failed_with_rollback'
                except Exception as rollback_error:
                    logger.error(f"Rollback also failed: {rollback_error}")
                    deployment_info['rollback_error'] = str(rollback_error)
            
            self.deployment_history.append(deployment_info)
            self._save_deployment_history()
            
            return {
                'status': deployment_info['status'],
                'deployment_id': deployment_id,
                'error': str(e),
                'details': deployment_info
            }
            
        finally:
            self.is_deploying = False
            self.current_deployment = None
    
    def rollback_to_previous(self, model_name: str) -> Dict[str, Any]:
        """Rollback to previous model version."""
        try:
            # Find last successful Развертывание
            successful_deployments = [
                d for d in reversed(self.deployment_history)
                if d['model_name'] == model_name and d['status'] == 'completed'
            ]
            
            if len(successful_deployments) < 2:
                return {'status': 'failed', 'reason': 'No previous version to rollback to'}
            
            previous_deployment = successful_deployments[1]  # Second most recent
            backup_info = previous_deployment.get('backup_result')
            
            if not backup_info:
                return {'status': 'failed', 'reason': 'No backup information found'}
            
            rollback_result = self._rollback_deployment(model_name, backup_info)
            
            # Логирование rollback
            rollback_info = {
                'deployment_id': f"rollback_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_name': model_name,
                'action': 'manual_rollback',
                'target_deployment': previous_deployment['deployment_id'],
                'rollback_result': rollback_result,
                'timestamp': datetime.now(),
                'status': 'completed' if rollback_result['success'] else 'failed'
            }
            
            self.deployment_history.append(rollback_info)
            self._save_deployment_history()
            
            return {
                'status': 'completed' if rollback_result['success'] else 'failed',
                'details': rollback_info
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            'is_deploying': self.is_deploying,
            'current_deployment': self.current_deployment,
            'last_deployment': self.deployment_history[-1] if self.deployment_history else None,
            'deployments_count': len(self.deployment_history)
        }
    
    def get_deployment_history(self, model_name: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get deployment history."""
        history = self.deployment_history
        
        if model_name:
            history = [d for d in history if d.get('model_name') == model_name]
        
        return list(reversed(history))[:limit]
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get available versions of a model."""
        versions = []
        
        # Проверка Продакшн
        prod_path = self.deployment_path / f"{model_name}.pkl"
        if prod_path.exists():
            stat = prod_path.stat()
            versions.append({
                'version': 'production',
                'path': str(prod_path),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'status': 'active'
            })
        
        # Проверка backups
        backup_pattern = f"{model_name}_backup_*.pkl"
        for backup_file in self.backup_path.glob(backup_pattern):
            stat = backup_file.stat()
            versions.append({
                'version': backup_file.stem.replace(f"{model_name}_backup_", ""),
                'path': str(backup_file),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'status': 'backup'
            })
        
        return sorted(versions, key=lambda x: x['modified'], reverse=True)
    
    def _validate_model(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """Validate model before deployment."""
        try:
            model_path = Path(model_path)
            
            # Проверка if file exists
            if not model_path.exists():
                return {'valid': False, 'reason': 'Model file does not exist'}
            
            # Проверка file size
            file_size = model_path.stat().st_size
            if file_size == 0:
                return {'valid': False, 'reason': 'Model file is empty'}
            
            # Try to Загрузка the Модель
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Basic Модель validation
            if not hasattr(model, 'recommend_for_user'):
                return {'valid': False, 'reason': 'Model does not have recommend_for_user method'}
            
            # Additional Модель-specific validation could go here
            
            return {
                'valid': True,
                'file_size': file_size,
                'model_type': type(model).__name__,
                'validation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'valid': False, 'reason': f'Model validation error: {str(e)}'}
    
    def _backup_current_model(self, model_name: str) -> Dict[str, Any]:
        """Backup current production model."""
        try:
            current_model_path = self.deployment_path / f"{model_name}.pkl"
            
            if not current_model_path.exists():
                return {'success': True, 'reason': 'No current model to backup'}
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{model_name}_backup_{timestamp}.pkl"
            backup_path = self.backup_path / backup_filename
            
            shutil.copy2(current_model_path, backup_path)
            
            return {
                'success': True,
                'backup_path': str(backup_path),
                'original_path': str(current_model_path),
                'backup_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _blue_green_deployment(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """Perform blue-green deployment."""
        try:
            # In blue-green Развертывание, we prepare the new Версия
            # and then switch atomically
            
            staging_path = self.deployment_path / f"{model_name}_staging.pkl"
            production_path = self.deployment_path / f"{model_name}.pkl"
            
            # Copy new Модель to Тестовая среда
            shutil.copy2(model_path, staging_path)
            
            # Atomic switch
            if production_path.exists():
                old_path = self.deployment_path / f"{model_name}_old.pkl"
                production_path.rename(old_path)
            
            staging_path.rename(production_path)
            
            # Очистка up old file
            old_path = self.deployment_path / f"{model_name}_old.pkl"
            if old_path.exists():
                old_path.unlink()
            
            return {
                'success': True,
                'strategy': 'blue_green',
                'deployment_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _canary_deployment(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """Perform canary deployment."""
        # For canary Развертывание, we would gradually roll out to a percentage of traffic
        # For now, we'll implement it as a blue-green Развертывание
        # In a real Система, this would involve Загрузка balancer Конфигурация
        
        return self._blue_green_deployment(model_name, model_path)
    
    def _immediate_deployment(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """Perform immediate deployment."""
        try:
            production_path = self.deployment_path / f"{model_name}.pkl"
            shutil.copy2(model_path, production_path)
            
            return {
                'success': True,
                'strategy': 'immediate',
                'deployment_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Immediate deployment failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _perform_health_check(self, model_name: str, timeout: int = 300) -> Dict[str, Any]:
        """Perform health check on deployed model."""
        try:
            start_time = time.time()
            
            # Загрузка the deployed Модель
            model_path = self.deployment_path / f"{model_name}.pkl"
            
            if not model_path.exists():
                return {'healthy': False, 'reason': 'Deployed model file not found'}
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Тестирование basic functionality
            try:
                # Try to get recommendations for a Тестирование Пользователь
                test_recommendations = model.recommend_for_user(user_idx=0, n_recommendations=5)
                
                if not test_recommendations:
                    return {'healthy': False, 'reason': 'Model returned empty recommendations'}
                
            except Exception as e:
                return {'healthy': False, 'reason': f'Model functionality test failed: {str(e)}'}
            
            # Call external health Проверка if available
            if self.health_check_callback:
                try:
                    external_result = self.health_check_callback(model_name)
                    if not external_result.get('healthy', True):
                        return {
                            'healthy': False,
                            'reason': f"External health check failed: {external_result.get('reason', 'Unknown')}"
                        }
                except Exception as e:
                    logger.warning(f"External health check failed: {e}")
            
            health_check_time = time.time() - start_time
            
            return {
                'healthy': True,
                'check_duration': health_check_time,
                'test_recommendations_count': len(test_recommendations),
                'check_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'healthy': False, 'reason': f'Health check error: {str(e)}'}
    
    def _rollback_deployment(self, model_name: str, backup_info: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback to previous model version."""
        try:
            if not backup_info.get('success', False):
                return {'success': False, 'reason': 'No valid backup available'}
            
            backup_path = Path(backup_info['backup_path'])
            production_path = self.deployment_path / f"{model_name}.pkl"
            
            if not backup_path.exists():
                return {'success': False, 'reason': 'Backup file not found'}
            
            # Copy Резервное копирование to Продакшн
            shutil.copy2(backup_path, production_path)
            
            return {
                'success': True,
                'backup_path': str(backup_path),
                'rollback_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Default Конфигурация
        default_config = {
            'deployment_strategies': ['blue_green', 'canary', 'immediate'],
            'default_strategy': 'blue_green',
            'health_check_timeout': 300,
            'auto_rollback': True,
            'backup_retention_days': 30,
            'max_backup_versions': 10
        }
        
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict[str, Any] = None):
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _save_deployment_history(self):
        """Save deployment history to file."""
        try:
            history_file = self.deployment_path.parent / "deployment_history.json"
            
            # Преобразование to serializable Форматирование
            serializable_history = []
            for entry in self.deployment_history:
                serializable_entry = entry.copy()
                # Преобразование datetime objects to strings
                for key, value in serializable_entry.items():
                    if isinstance(value, datetime):
                        serializable_entry[key] = value.isoformat()
                
                serializable_history.append(serializable_entry)
            
            with open(history_file, 'w') as f:
                json.dump(serializable_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving deployment history: {e}")
    
    def cleanup_old_backups(self, model_name: str = None):
        """Clean up old backup files."""
        try:
            retention_days = self.config.get('backup_retention_days', 30)
            max_versions = self.config.get('max_backup_versions', 10)
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            if model_name:
                pattern = f"{model_name}_backup_*.pkl"
            else:
                pattern = "*_backup_*.pkl"
            
            backup_files = list(self.backup_path.glob(pattern))
            
            # Сортировка by modification time
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Удаление old files
            removed_count = 0
            for backup_file in backup_files:
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                
                # Удаление if too old or beyond max versions
                if (file_time < cutoff_date or 
                    backup_files.index(backup_file) >= max_versions):
                    backup_file.unlink()
                    removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old backup files")
            
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}") 