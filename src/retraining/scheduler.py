"""
Retraining Scheduler

Manages automatic model retraining schedules and triggers.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import schedule
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """Types of retraining triggers."""
    SCHEDULED = "scheduled"
    METRIC_DEGRADATION = "metric_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    ERROR_RATE = "error_rate"

class RetrainingScheduler:
    """
    Manages automatic model retraining schedules and triggers.
    
    Features:
    - Multiple trigger types (time-based, metric-based, manual)
    - Configurable schedules
    - Trigger history and logging
    - Integration with metrics monitor and model trainer
    """
    
    def __init__(
        self,
        config_path: str = "config/retraining_config.json",
        logs_path: str = "logs/retraining"
    ):
        self.config_path = Path(config_path)
        self.logs_path = Path(logs_path)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Загрузка Конфигурация
        self.config = self._load_config()
        
        # Состояние
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.retraining_callback: Optional[Callable] = None
        self.metrics_monitor = None
        self.model_trainer = None
        
        # History
        self.trigger_history: List[Dict[str, Any]] = []
        
        # Настройка schedules
        self._setup_schedules()
        
        logger.info("RetrainingScheduler initialized")
    
    def set_retraining_callback(self, callback: Callable):
        """Set callback function for triggering retraining."""
        self.retraining_callback = callback
    
    def set_metrics_monitor(self, monitor):
        """Set metrics monitor instance."""
        self.metrics_monitor = monitor
        if monitor:
            monitor.add_retraining_callback(self._on_metric_trigger)
    
    def set_model_trainer(self, trainer):
        """Set model trainer instance."""
        self.model_trainer = trainer
    
    def start(self):
        """Start the retraining scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Retraining scheduler started")
    
    def stop(self):
        """Stop the retraining scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        
        logger.info("Retraining scheduler stopped")
    
    def trigger_manual_retraining(
        self,
        reason: str = "Manual trigger",
        models: List[str] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Manually trigger model retraining."""
        trigger_info = {
            'trigger_type': TriggerType.MANUAL.value,
            'reason': reason,
            'models': models,
            'priority': priority,
            'timestamp': datetime.now().isoformat(),
            'triggered_by': 'manual'
        }
        
        return self._execute_trigger(trigger_info)
    
    def add_schedule(
        self,
        name: str,
        schedule_type: str,
        schedule_value: str,
        models: List[str] = None,
        enabled: bool = True
    ):
        """Add a new retraining schedule."""
        if 'schedules' not in self.config:
            self.config['schedules'] = {}
        
        self.config['schedules'][name] = {
            'type': schedule_type,
            'value': schedule_value,
            'models': models,
            'enabled': enabled,
            'created_at': datetime.now().isoformat()
        }
        
        self._save_config()
        self._setup_schedules()
        
        logger.info(f"Added schedule: {name}")
    
    def remove_schedule(self, name: str):
        """Remove a retraining schedule."""
        if 'schedules' in self.config and name in self.config['schedules']:
            del self.config['schedules'][name]
            self._save_config()
            self._setup_schedules()
            logger.info(f"Removed schedule: {name}")
        else:
            logger.warning(f"Schedule not found: {name}")
    
    def get_schedules(self) -> Dict[str, Any]:
        """Get all configured schedules."""
        return self.config.get('schedules', {})
    
    def get_trigger_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get trigger history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            trigger for trigger in self.trigger_history
            if datetime.fromisoformat(trigger['timestamp']) >= cutoff_time
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'is_running': self.is_running,
            'schedules_count': len(self.config.get('schedules', {})),
            'active_schedules': len([
                s for s in self.config.get('schedules', {}).values()
                if s.get('enabled', True)
            ]),
            'triggers_last_24h': len(self.get_trigger_history(24)),
            'last_trigger': self.trigger_history[-1] if self.trigger_history else None,
            'next_scheduled_run': self._get_next_scheduled_run()
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load retraining configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info("Loaded retraining configuration")
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Default Конфигурация
        default_config = {
            'schedules': {
                'daily_retrain': {
                    'type': 'daily',
                    'value': '02:00',
                    'models': ['popularity', 'als'],
                    'enabled': True
                },
                'weekly_full_retrain': {
                    'type': 'weekly',
                    'value': 'sunday:03:00',
                    'models': None,  # All models
                    'enabled': True
                }
            },
            'triggers': {
                'metric_degradation_threshold': 0.15,
                'error_rate_threshold': 0.1,
                'min_time_between_triggers': 3600  # 1 hour
            },
            'retraining': {
                'max_concurrent_trainings': 1,
                'validation_split': 0.2,
                'auto_deploy_threshold': 0.05  # 5% improvement
            }
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
    
    def _setup_schedules(self):
        """Setup scheduled triggers."""
        # Clear existing schedules
        schedule.clear()
        
        schedules_config = self.config.get('schedules', {})
        
        for name, schedule_config in schedules_config.items():
            if not schedule_config.get('enabled', True):
                continue
            
            schedule_type = schedule_config['type']
            schedule_value = schedule_config['value']
            models = schedule_config.get('models')
            
            try:
                if schedule_type == 'daily':
                    # Форматирование: "HH:MM"
                    schedule.every().day.at(schedule_value).do(
                        self._scheduled_trigger, name, models
                    )
                elif schedule_type == 'weekly':
                    # Форматирование: "day:HH:MM"
                    parts = schedule_value.split(':')
                    if len(parts) == 3:
                        day, hour, minute = parts
                        time_str = f"{hour}:{minute}"
                    else:
                        time_str = schedule_value
                    schedule.every().week.at(time_str).do(
                        self._scheduled_trigger, name, models
                    )
                elif schedule_type == 'hourly':
                    # Форматирование: "N" (every N hours)
                    hours = int(schedule_value)
                    schedule.every(hours).hours.do(
                        self._scheduled_trigger, name, models
                    )
                
                logger.info(f"Scheduled: {name} ({schedule_type}: {schedule_value})")
                
            except Exception as e:
                logger.error(f"Error setting up schedule {name}: {e}")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    def _scheduled_trigger(self, schedule_name: str, models: List[str] = None):
        """Handle scheduled trigger."""
        trigger_info = {
            'trigger_type': TriggerType.SCHEDULED.value,
            'reason': f'Scheduled trigger: {schedule_name}',
            'models': models,
            'schedule_name': schedule_name,
            'timestamp': datetime.now().isoformat(),
            'triggered_by': 'scheduler'
        }
        
        self._execute_trigger(trigger_info)
    
    def _on_metric_trigger(self, metric_trigger_info: Dict[str, Any]):
        """Handle metric-based trigger."""
        trigger_info = {
            'trigger_type': TriggerType.METRIC_DEGRADATION.value,
            'reason': metric_trigger_info['reason'],
            'metric_name': metric_trigger_info['metric_name'],
            'metric_value': metric_trigger_info['metric_value'],
            'timestamp': metric_trigger_info['timestamp'],
            'triggered_by': 'metrics_monitor'
        }
        
        self._execute_trigger(trigger_info)
    
    def _execute_trigger(self, trigger_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a retraining trigger."""
        # Проверка if we should throttle triggers
        if not self._should_execute_trigger(trigger_info):
            logger.info(f"Trigger throttled: {trigger_info['reason']}")
            return {'status': 'throttled', 'reason': 'Too soon since last trigger'}
        
        # Логирование trigger
        self.trigger_history.append(trigger_info)
        self._save_trigger_log(trigger_info)
        
        logger.info(f"Executing retraining trigger: {trigger_info['reason']}")
        
        try:
            # Execute retraining Колбэк
            if self.retraining_callback:
                result = self.retraining_callback(trigger_info)
                
                # Обновление trigger with result
                trigger_info['result'] = result
                trigger_info['status'] = 'completed'
                
                return result
            else:
                logger.warning("No retraining callback configured")
                return {'status': 'failed', 'reason': 'No callback configured'}
                
        except Exception as e:
            logger.error(f"Error executing trigger: {e}")
            trigger_info['status'] = 'failed'
            trigger_info['error'] = str(e)
            return {'status': 'failed', 'error': str(e)}
    
    def _should_execute_trigger(self, trigger_info: Dict[str, Any]) -> bool:
        """Check if trigger should be executed (throttling)."""
        min_interval = self.config.get('triggers', {}).get('min_time_between_triggers', 3600)
        
        if not self.trigger_history:
            return True
        
        last_trigger_time = datetime.fromisoformat(self.trigger_history[-1]['timestamp'])
        current_time = datetime.now()
        
        time_since_last = (current_time - last_trigger_time).total_seconds()
        
        # Allow manual triggers to bypass throttling
        if trigger_info.get('trigger_type') == TriggerType.MANUAL.value:
            return True
        
        return time_since_last >= min_interval
    
    def _save_trigger_log(self, trigger_info: Dict[str, Any]):
        """Save trigger to log file."""
        try:
            log_file = self.logs_path / f"triggers_{datetime.now().strftime('%Y%m')}.jsonl"
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(trigger_info, default=str) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving trigger log: {e}")
    
    def _get_next_scheduled_run(self) -> Optional[str]:
        """Get next scheduled run time."""
        try:
            next_run = schedule.next_run()
            return next_run.isoformat() if next_run else None
        except Exception:
            return None
    
    def update_schedule(self, name: str, updates: Dict[str, Any]):
        """Update an existing schedule."""
        if 'schedules' not in self.config:
            self.config['schedules'] = {}
        
        if name in self.config['schedules']:
            self.config['schedules'][name].update(updates)
            self._save_config()
            self._setup_schedules()
            logger.info(f"Updated schedule: {name}")
        else:
            logger.warning(f"Schedule not found: {name}")
    
    def enable_schedule(self, name: str):
        """Enable a schedule."""
        self.update_schedule(name, {'enabled': True})
    
    def disable_schedule(self, name: str):
        """Disable a schedule."""
        self.update_schedule(name, {'enabled': False})
    
    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Get performance report for recent triggers."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        recent_triggers = [
            trigger for trigger in self.trigger_history
            if datetime.fromisoformat(trigger['timestamp']) >= cutoff_time
        ]
        
        if not recent_triggers:
            return {'message': 'No triggers in specified period'}
        
        # Анализ triggers
        trigger_types = {}
        success_count = 0
        total_count = len(recent_triggers)
        
        for trigger in recent_triggers:
            trigger_type = trigger.get('trigger_type', 'unknown')
            trigger_types[trigger_type] = trigger_types.get(trigger_type, 0) + 1
            
            if trigger.get('status') == 'completed':
                success_count += 1
        
        return {
            'period_days': days,
            'total_triggers': total_count,
            'successful_triggers': success_count,
            'success_rate': success_count / total_count if total_count > 0 else 0,
            'trigger_types': trigger_types,
            'avg_triggers_per_day': total_count / days,
            'recent_triggers': recent_triggers[-5:]  # Last 5 triggers
        } 