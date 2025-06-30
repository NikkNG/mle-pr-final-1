"""
Конфигурация для рекомендательной системы e-commerce
"""
import os
from typing import Optional


class Config:
    """Базовая конфигурация"""
    
    # Конфигурация базы данных
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ecommerce_recommender.db")
    DATABASE_HOST: str = os.getenv("DATABASE_HOST", "localhost")
    DATABASE_PORT: int = int(os.getenv("DATABASE_PORT", "5432"))
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "ecommerce_recommender")
    DATABASE_USER: str = os.getenv("DATABASE_USER", "username")
    DATABASE_PASSWORD: str = os.getenv("DATABASE_PASSWORD", "password")

    # Конфигурация MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "ecommerce-recommender")
    MLFLOW_ARTIFACT_ROOT: str = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")

    # Конфигурация API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Конфигурация модели
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./data/models/")
    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", "hybrid_recommender")
    RECOMMENDATION_COUNT: int = int(os.getenv("RECOMMENDATION_COUNT", "10"))

    # Конфигурация данных
    DATA_PATH: str = os.getenv("DATA_PATH", "./data/raw/")
    PROCESSED_DATA_PATH: str = os.getenv("PROCESSED_DATA_PATH", "./data/processed/")

    # Конфигурация мониторинга
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "8080"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Конфигурация Airflow
    AIRFLOW_HOME: str = os.getenv("AIRFLOW_HOME", "./airflow")
    AIRFLOW_DAGS_FOLDER: str = os.getenv("AIRFLOW_DAGS_FOLDER", "./airflow/dags")
    AIRFLOW_LOGS_FOLDER: str = os.getenv("AIRFLOW_LOGS_FOLDER", "./airflow/logs")

    # Безопасность
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    API_KEY: Optional[str] = os.getenv("API_KEY")

    # Производительность
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1000"))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "8"))

    # Флаги функций
    ENABLE_MONITORING: bool = os.getenv("ENABLE_MONITORING", "True").lower() == "true"
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "True").lower() == "true"
    ENABLE_A_B_TESTING: bool = os.getenv("ENABLE_A_B_TESTING", "False").lower() == "true"


class DevelopmentConfig(Config):
    """Конфигурация для разработки"""
    DEBUG = True
    MLFLOW_TRACKING_URI = "http://localhost:5000"


class ProductionConfig(Config):
    """Конфигурация для продакшена"""
    DEBUG = False
    API_WORKERS = 8


class TestingConfig(Config):
    """Конфигурация для тестирования"""
    DATABASE_URL = "sqlite:///:memory:"
    TESTING = True


# Автоматический выбор конфигурации на основе переменной окружения
def get_config() -> Config:
    """Получить конфигурацию на основе переменной окружения"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


# Глобальный объект конфигурации
config = get_config() 