"""
Alerting system for the recommendation system monitoring.

Monitors metrics in real-time and triggers alerts when thresholds are breached.
Supports multiple notification channels and alert escalation.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

from .metrics import metrics_collector, MetricsCollector

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    metric_name: str
    threshold: float
    operator: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    severity: AlertSeverity
    message: str
    evaluation_window: int = 300  # seconds
    min_samples: int = 1
    cooldown_period: int = 600  # seconds before re-alerting
    enabled: bool = True


@dataclass
class Alert:
    """Active alert instance."""
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    metric_name: str
    current_value: float
    threshold: float
    first_triggered: datetime
    last_triggered: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    notification_count: int = 0


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
    
    def send_notification(self, alert: Alert) -> bool:
        """Send notification for an alert. Return True if successful."""
        raise NotImplementedError


class LogNotificationChannel(NotificationChannel):
    """Log-based notification channel."""
    
    def send_notification(self, alert: Alert) -> bool:
        try:
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }[alert.severity]
            
            logger.log(
                log_level,
                f"ALERT [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message} "
                f"(Current: {alert.current_value}, Threshold: {alert.threshold})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send log notification: {e}")
            return False


class FileNotificationChannel(NotificationChannel):
    """File-based notification channel."""
    
    def send_notification(self, alert: Alert) -> bool:
        try:
            alerts_dir = Path(self.config.get("alerts_dir", "logs/alerts"))
            alerts_dir.mkdir(parents=True, exist_ok=True)
            
            alert_file = alerts_dir / f"alert_{alert.id}.json"
            
            alert_data = {
                "alert": asdict(alert),
                "timestamp": datetime.now().isoformat()
            }
            
            # Преобразование datetime objects to strings for JSON serialization
            alert_data["alert"]["first_triggered"] = alert.first_triggered.isoformat()
            alert_data["alert"]["last_triggered"] = alert.last_triggered.isoformat()
            if alert.resolved_at:
                alert_data["alert"]["resolved_at"] = alert.resolved_at.isoformat()
            if alert.acknowledged_at:
                alert_data["alert"]["acknowledged_at"] = alert.acknowledged_at.isoformat()
            
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to send file notification: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Webhook-based notification channel."""
    
    def send_notification(self, alert: Alert) -> bool:
        try:
            import requests
            
            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
            
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "timestamp": alert.last_triggered.isoformat()
            }
            
            headers = {"Content-Type": "application/json"}
            timeout = self.config.get("timeout", 10)
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class AlertManager:
    """
    Manages alert rules, evaluates conditions, and sends notifications.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._evaluation_thread = None
        self._stop_evaluation = threading.Event()
        self._evaluation_interval = 30  # seconds
        
        # Alert history
        self.alert_history: List[Alert] = []
        self.max_history_size = 1000
        
        # Настройка default notification channels
        self._setup_default_channels()
        
        # Настройка default alert rules
        self._setup_default_rules()
    
    def _setup_default_channels(self):
        """Setup default notification channels."""
        # Логирование channel
        self.add_notification_channel(
            "log",
            LogNotificationChannel("log", {"enabled": True})
        )
        
        # File channel
        self.add_notification_channel(
            "file",
            FileNotificationChannel("file", {
                "enabled": True,
                "alerts_dir": "logs/alerts"
            })
        )
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                metric_name="api_error_rate",
                threshold=0.05,  # 5%
                operator="gt",
                severity=AlertSeverity.CRITICAL,
                message="High API error rate detected",
                evaluation_window=300,
                min_samples=5
            ),
            AlertRule(
                name="slow_response_time",
                metric_name="api_avg_response_time",
                threshold=1.0,  # 1 second
                operator="gt",
                severity=AlertSeverity.WARNING,
                message="Slow API response time detected",
                evaluation_window=300,
                min_samples=10
            ),
            AlertRule(
                name="low_ctr",
                metric_name="business_ctr",
                threshold=0.005,  # 0.5%
                operator="lt",
                severity=AlertSeverity.WARNING,
                message="Low recommendation click-through rate",
                evaluation_window=1800,  # 30 minutes
                min_samples=100
            ),
            AlertRule(
                name="low_catalog_coverage",
                metric_name="business_catalog_coverage",
                threshold=0.1,  # 10%
                operator="lt",
                severity=AlertSeverity.WARNING,
                message="Low catalog coverage in recommendations",
                evaluation_window=3600,  # 1 hour
                min_samples=1
            ),
            AlertRule(
                name="no_requests",
                metric_name="api_request_count",
                threshold=1,
                operator="lt",
                severity=AlertSeverity.CRITICAL,
                message="No API requests received",
                evaluation_window=600,  # 10 minutes
                min_samples=1
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def add_notification_channel(self, name: str, channel: NotificationChannel):
        """Add a notification channel."""
        with self._lock:
            self.notification_channels[name] = channel
            logger.info(f"Added notification channel: {name}")
    
    def remove_notification_channel(self, name: str):
        """Remove a notification channel."""
        with self._lock:
            if name in self.notification_channels:
                del self.notification_channels[name]
                logger.info(f"Removed notification channel: {name}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self._lock:
            self.alert_rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        with self._lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
    
    def start_monitoring(self):
        """Start the alert evaluation thread."""
        if self._evaluation_thread is None or not self._evaluation_thread.is_alive():
            self._stop_evaluation.clear()
            self._evaluation_thread = threading.Thread(
                target=self._evaluation_loop,
                daemon=True
            )
            self._evaluation_thread.start()
            logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop the alert evaluation thread."""
        self._stop_evaluation.set()
        if self._evaluation_thread and self._evaluation_thread.is_alive():
            self._evaluation_thread.join(timeout=10)
            logger.info("Alert monitoring stopped")
    
    def _evaluation_loop(self):
        """Main evaluation loop for checking alert conditions."""
        while not self._stop_evaluation.wait(self._evaluation_interval):
            try:
                self._evaluate_all_rules()
                self._cleanup_resolved_alerts()
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
    
    def _evaluate_all_rules(self):
        """Evaluate all enabled alert rules."""
        current_metrics = self.metrics_collector.get_current_metrics()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                self._evaluate_rule(rule, current_metrics)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    def _evaluate_rule(self, rule: AlertRule, current_metrics: Dict[str, Any]):
        """Evaluate a single alert rule."""
        # Extract Метрика value based on rule
        metric_value = self._extract_metric_value(rule.metric_name, current_metrics)
        
        if metric_value is None:
            return
        
        # Проверка if condition is met
        condition_met = self._check_condition(metric_value, rule.threshold, rule.operator)
        
        alert_id = f"{rule.name}_{rule.metric_name}"
        existing_alert = self.active_alerts.get(alert_id)
        
        if condition_met:
            if existing_alert:
                # Обновление existing alert
                existing_alert.last_triggered = datetime.now()
                existing_alert.current_value = metric_value
                existing_alert.notification_count += 1
            else:
                # Создание new alert
                new_alert = Alert(
                    id=alert_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    message=rule.message,
                    metric_name=rule.metric_name,
                    current_value=metric_value,
                    threshold=rule.threshold,
                    first_triggered=datetime.now(),
                    last_triggered=datetime.now(),
                    notification_count=1
                )
                
                with self._lock:
                    self.active_alerts[alert_id] = new_alert
                
                # Отправка notifications
                self._send_notifications(new_alert)
                
                logger.warning(f"Alert triggered: {rule.name}")
        
        else:
            # Condition not met, resolve alert if active
            if existing_alert and existing_alert.status == AlertStatus.ACTIVE:
                existing_alert.status = AlertStatus.RESOLVED
                existing_alert.resolved_at = datetime.now()
                
                # Move to history
                self.alert_history.append(existing_alert)
                if len(self.alert_history) > self.max_history_size:
                    self.alert_history.pop(0)
                
                with self._lock:
                    del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {rule.name}")
    
    def _extract_metric_value(self, metric_name: str, current_metrics: Dict[str, Any]) -> Optional[float]:
        """Extract metric value from current metrics."""
        try:
            if metric_name.startswith("api_"):
                metric_key = metric_name[4:]  # Remove 'api_' prefix
                return current_metrics.get("api_metrics", {}).get(metric_key)
            
            elif metric_name.startswith("business_"):
                metric_key = metric_name[9:]  # Remove 'business_' prefix
                return current_metrics.get("business_metrics", {}).get(metric_key)
            
            else:
                # Direct Метрика name
                return current_metrics.get(metric_name)
        
        except Exception as e:
            logger.error(f"Error extracting metric {metric_name}: {e}")
            return None
    
    def _check_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Check if alert condition is met."""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        elif operator == "eq":
            return abs(value - threshold) < 1e-9
        else:
            logger.error(f"Unknown operator: {operator}")
            return False
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for channel_name, channel in self.notification_channels.items():
            if not channel.enabled:
                continue
            
            try:
                success = channel.send_notification(alert)
                if success:
                    logger.debug(f"Notification sent via {channel_name} for alert {alert.id}")
                else:
                    logger.warning(f"Failed to send notification via {channel_name} for alert {alert.id}")
            except Exception as e:
                logger.error(f"Error sending notification via {channel_name}: {e}")
    
    def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Удаление old alerts from history
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.resolved_at and alert.resolved_at > cutoff_time
        ]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.first_triggered > cutoff_time
        ]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
                self.active_alerts[alert_id].acknowledged_at = datetime.now()
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
            return False
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        with self._lock:
            active_alerts = list(self.active_alerts.values())
            
            summary = {
                "total_active": len(active_alerts),
                "by_severity": {
                    "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                    "warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
                    "info": len([a for a in active_alerts if a.severity == AlertSeverity.INFO]),
                },
                "by_status": {
                    "active": len([a for a in active_alerts if a.status == AlertStatus.ACTIVE]),
                    "acknowledged": len([a for a in active_alerts if a.status == AlertStatus.ACKNOWLEDGED]),
                },
                "total_rules": len(self.alert_rules),
                "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
                "notification_channels": len(self.notification_channels),
                "timestamp": datetime.now().isoformat()
            }
            
            return summary


# Global alert manager instance
alert_manager = AlertManager(metrics_collector) 