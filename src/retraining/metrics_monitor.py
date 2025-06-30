"""
Metrics Monitor for Model Performance Tracking

Monitors model performance metrics and determines when retraining is needed.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MetricThreshold:
    """Configuration for metric thresholds."""
    metric_name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    trend_window: int = 10  # Number of measurements for trend analysis
    degradation_threshold: float = 0.1  # 10% degradation triggers retraining

@dataclass
class MetricMeasurement:
    """Single metric measurement."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any]

class MetricsMonitor:
    """
    Monitors model performance metrics and triggers retraining when needed.
    
    Features:
    - Real-time metric tracking
    - Threshold-based alerting
    - Trend analysis
    - Performance degradation detection
    - Integration with MLflow
    """
    
    def __init__(
        self,
        metrics_config: Dict[str, MetricThreshold],
        storage_path: str = "logs/metrics",
        check_interval: int = 300,  # 5 minutes
        history_retention_days: int = 30
    ):
        self.metrics_config = metrics_config
        self.storage_path = Path(storage_path)
        self.check_interval = check_interval
        self.history_retention_days = history_retention_days
        
        # Создание storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Метрика storage
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.current_metrics: Dict[str, float] = {}
        
        # Мониторинг Состояние
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks for retraining triggers
        self.retraining_callbacks: List[Callable] = []
        
        # Производительность baselines
        self.baselines: Dict[str, float] = {}
        
        logger.info(f"MetricsMonitor initialized with {len(metrics_config)} metrics")
    
    def add_retraining_callback(self, callback: Callable):
        """Add callback function to be called when retraining is triggered."""
        self.retraining_callbacks.append(callback)
    
    def set_baseline(self, metric_name: str, value: float):
        """Set performance baseline for a metric."""
        self.baselines[metric_name] = value
        logger.info(f"Baseline set for {metric_name}: {value}")
    
    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a new metric measurement."""
        if metadata is None:
            metadata = {}
        
        measurement = MetricMeasurement(
            timestamp=datetime.now(),
            value=value,
            metadata=metadata
        )
        
        # Сохранение in memory
        self.metrics_history[metric_name].append(measurement)
        self.current_metrics[metric_name] = value
        
        # Сохранение to disk
        self._save_measurement(metric_name, measurement)
        
        # Проверка if retraining is needed
        if self.is_monitoring:
            self._check_retraining_triggers(metric_name, value)
        
        logger.debug(f"Recorded metric {metric_name}: {value}")
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Metrics monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Metrics monitoring stopped")
    
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for a metric over specified time period."""
        if metric_name not in self.metrics_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_measurements = [
            m for m in self.metrics_history[metric_name]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_measurements:
            return {}
        
        values = [m.value for m in recent_measurements]
        
        return {
            'metric_name': metric_name,
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'latest': values[-1],
            'trend': self._calculate_trend(values),
            'baseline_comparison': self._compare_to_baseline(metric_name, np.mean(values))
        }
    
    def get_all_metrics_summary(self, hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """Get summary for all tracked metrics."""
        return {
            metric_name: self.get_metric_summary(metric_name, hours)
            for metric_name in self.metrics_history.keys()
        }
    
    def check_model_health(self) -> Dict[str, Any]:
        """Comprehensive model health check."""
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'alerts': [],
            'recommendations': []
        }
        
        for metric_name, config in self.metrics_config.items():
            metric_health = self._check_metric_health(metric_name, config)
            health_status['metrics'][metric_name] = metric_health
            
            if metric_health['status'] != 'healthy':
                health_status['overall_status'] = 'degraded'
                health_status['alerts'].append(metric_health['alert'])
        
        # Добавление recommendations
        if health_status['overall_status'] == 'degraded':
            health_status['recommendations'].append('Consider model retraining')
        
        return health_status
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Perform periodic health checks
                health_status = self.check_model_health()
                
                # Логирование health Статус
                if health_status['overall_status'] != 'healthy':
                    logger.warning(f"Model health degraded: {health_status['alerts']}")
                
                # Очистка old data
                self._cleanup_old_data()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_retraining_triggers(self, metric_name: str, value: float):
        """Check if retraining should be triggered based on metric value."""
        if metric_name not in self.metrics_config:
            return
        
        config = self.metrics_config[metric_name]
        should_retrain = False
        reason = ""
        
        # Проверка absolute thresholds
        if config.min_value is not None and value < config.min_value:
            should_retrain = True
            reason = f"{metric_name} below minimum threshold: {value} < {config.min_value}"
        
        if config.max_value is not None and value > config.max_value:
            should_retrain = True
            reason = f"{metric_name} above maximum threshold: {value} > {config.max_value}"
        
        # Проверка trend-based degradation
        if len(self.metrics_history[metric_name]) >= config.trend_window:
            recent_values = [
                m.value for m in list(self.metrics_history[metric_name])[-config.trend_window:]
            ]
            
            if metric_name in self.baselines:
                recent_avg = np.mean(recent_values)
                baseline = self.baselines[metric_name]
                degradation = (baseline - recent_avg) / baseline
                
                if degradation > config.degradation_threshold:
                    should_retrain = True
                    reason = f"{metric_name} degraded by {degradation:.1%} from baseline"
        
        if should_retrain:
            logger.warning(f"Retraining trigger: {reason}")
            self._trigger_retraining(reason, metric_name, value)
    
    def _trigger_retraining(self, reason: str, metric_name: str, value: float):
        """Trigger retraining callbacks."""
        trigger_info = {
            'reason': reason,
            'metric_name': metric_name,
            'metric_value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        for callback in self.retraining_callbacks:
            try:
                callback(trigger_info)
            except Exception as e:
                logger.error(f"Error in retraining callback: {e}")
    
    def _check_metric_health(self, metric_name: str, config: MetricThreshold) -> Dict[str, Any]:
        """Check health status of a specific metric."""
        if metric_name not in self.metrics_history:
            return {
                'status': 'unknown',
                'message': 'No data available',
                'alert': f"No data for metric {metric_name}"
            }
        
        recent_measurements = list(self.metrics_history[metric_name])[-config.trend_window:]
        if not recent_measurements:
            return {
                'status': 'unknown',
                'message': 'Insufficient data',
                'alert': f"Insufficient data for metric {metric_name}"
            }
        
        latest_value = recent_measurements[-1].value
        recent_values = [m.value for m in recent_measurements]
        
        # Проверка thresholds
        if config.min_value is not None and latest_value < config.min_value:
            return {
                'status': 'critical',
                'message': f'Below minimum threshold: {latest_value} < {config.min_value}',
                'alert': f"Critical: {metric_name} below threshold"
            }
        
        if config.max_value is not None and latest_value > config.max_value:
            return {
                'status': 'critical',
                'message': f'Above maximum threshold: {latest_value} > {config.max_value}',
                'alert': f"Critical: {metric_name} above threshold"
            }
        
        # Проверка trend
        if metric_name in self.baselines and len(recent_values) >= 3:
            recent_avg = np.mean(recent_values)
            baseline = self.baselines[metric_name]
            degradation = (baseline - recent_avg) / baseline
            
            if degradation > config.degradation_threshold:
                return {
                    'status': 'warning',
                    'message': f'Performance degraded by {degradation:.1%}',
                    'alert': f"Warning: {metric_name} performance degraded"
                }
        
        return {
            'status': 'healthy',
            'message': 'Within normal parameters',
            'alert': None
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'improving'
        else:
            return 'degrading'
    
    def _compare_to_baseline(self, metric_name: str, current_value: float) -> Dict[str, Any]:
        """Compare current value to baseline."""
        if metric_name not in self.baselines:
            return {'status': 'no_baseline'}
        
        baseline = self.baselines[metric_name]
        change = (current_value - baseline) / baseline
        
        return {
            'baseline': baseline,
            'current': current_value,
            'change_percent': change * 100,
            'status': 'better' if change > 0 else 'worse' if change < -0.05 else 'similar'
        }
    
    def _save_measurement(self, metric_name: str, measurement: MetricMeasurement):
        """Save measurement to disk."""
        try:
            file_path = self.storage_path / f"{metric_name}.jsonl"
            
            record = {
                'timestamp': measurement.timestamp.isoformat(),
                'value': measurement.value,
                'metadata': measurement.metadata
            }
            
            with open(file_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving measurement: {e}")
    
    def _cleanup_old_data(self):
        """Remove old metric data beyond retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.history_retention_days)
        
        for metric_name in self.metrics_history:
            # Очистка memory storage
            while (self.metrics_history[metric_name] and 
                   self.metrics_history[metric_name][0].timestamp < cutoff_date):
                self.metrics_history[metric_name].popleft()
    
    def export_metrics_data(self, metric_name: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Export metric data as DataFrame for analysis."""
        file_path = self.storage_path / f"{metric_name}.jsonl"
        
        if not file_path.exists():
            return pd.DataFrame()
        
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    timestamp = datetime.fromisoformat(record['timestamp'])
                    if start_date <= timestamp <= end_date:
                        records.append(record)
                except Exception as e:
                    logger.warning(f"Error parsing metric record: {e}")
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')


def create_default_metrics_config() -> Dict[str, MetricThreshold]:
    """Create default metrics configuration for recommendation systems."""
    return {
        'precision_at_10': MetricThreshold(
            metric_name='precision_at_10',
            min_value=0.1,
            degradation_threshold=0.15
        ),
        'recall_at_10': MetricThreshold(
            metric_name='recall_at_10',
            min_value=0.05,
            degradation_threshold=0.15
        ),
        'ndcg_at_10': MetricThreshold(
            metric_name='ndcg_at_10',
            min_value=0.1,
            degradation_threshold=0.15
        ),
        'coverage': MetricThreshold(
            metric_name='coverage',
            min_value=0.1,
            degradation_threshold=0.2
        ),
        'diversity': MetricThreshold(
            metric_name='diversity',
            min_value=0.3,
            degradation_threshold=0.2
        ),
        'response_time': MetricThreshold(
            metric_name='response_time',
            max_value=0.5,  # 500ms
            degradation_threshold=0.3
        ),
        'error_rate': MetricThreshold(
            metric_name='error_rate',
            max_value=0.05,  # 5%
            degradation_threshold=0.5
        )
    } 