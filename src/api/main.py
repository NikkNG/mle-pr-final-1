"""
FastAPI service for recommendation system.
"""

import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Path, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Импорт схем и менеджера моделей
from .schemas import (
    RecommendationRequest, RecommendationResponse, RecommendationItem,
    SimilarItemsRequest, SimilarItemsResponse,
    PopularItemsRequest, PopularItemsResponse,
    HealthResponse, ErrorResponse,
    BatchRecommendationRequest, BatchRecommendationResponse,
    ModelsStatusResponse, ModelInfo
)
from .model_manager import ModelManager

# Импорт эндпоинтов для переобучения
from .retraining_endpoints import router as retraining_router, initialize_retraining_service
from ..retraining.retraining_service import create_retraining_service

# Импорт компонентов мониторинга
from ..monitoring.dashboard import monitoring_router
from ..monitoring.metrics import metrics_collector
from ..monitoring.alerting import alert_manager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальный экземпляр менеджера моделей
model_manager: Optional[ModelManager] = None

# Глобальный экземпляр сервиса переобучения
retraining_service_instance: Optional[Any] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    global model_manager, retraining_service_instance
    
    # Запуск
    logger.info("Starting recommendation service...")
    try:
        model_manager = ModelManager()
        success = model_manager.load_models()
        if not success:
            logger.error("Failed to load models")
            raise RuntimeError("Model loading failed")
        logger.info("Models loaded successfully")
        
        # Инициализация сервиса переобучения
        logger.info("Initializing retraining service...")
        retraining_service_instance = create_retraining_service()
        initialize_retraining_service(retraining_service_instance)
        retraining_service_instance.start()
        logger.info("Retraining service started")
        
        # Инициализация мониторинга
        logger.info("Starting monitoring services...")
        alert_manager.start_monitoring()
        logger.info("Monitoring services started")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Завершение работы
    logger.info("Shutting down recommendation service...")
    if retraining_service_instance:
        retraining_service_instance.stop()
        logger.info("Retraining service stopped")


# Создание FastAPI app
app = FastAPI(
    title="E-commerce Recommendation API",
    description="API service for product recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Добавление CORS Промежуточное ПО
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать разрешенные источники
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(retraining_router)
app.include_router(monitoring_router)


def get_model_manager() -> ModelManager:
    """Зависимость для получения экземпляра менеджера моделей."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return model_manager


async def log_request(request_type: str, user_id: Optional[int] = None, 
                     item_id: Optional[int] = None):
    """Логирование API запросов для мониторинга."""
    logger.info(f"API Request: {request_type}, user_id: {user_id}, item_id: {item_id}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Корневой эндпоинт."""
    return {
        "message": "E-commerce Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """Эндпоинт проверки состояния здоровья."""
    try:
        is_healthy = manager.is_healthy()
        status = "healthy" if is_healthy else "unhealthy"
        
        models_status = {}
        for model_name in manager.models.keys():
            models_status[model_name] = True
        
        return HealthResponse(
            status=status,
            version="1.0.0",
            models_loaded=models_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations(
    user_id: int = Path(..., gt=0, description="ID пользователя"),
    num_recommendations: int = Query(10, ge=1, le=100, description="Количество рекомендаций"),
    model: str = Query("hybrid", description="Модель для создания рекомендаций"),
    exclude_seen: bool = Query(True, description="Исключить товары, с которыми пользователь уже взаимодействовал"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    manager: ModelManager = Depends(get_model_manager)
):
    """Получить персонализированные рекомендации для пользователя."""
    try:
        # Логирование запроса
        background_tasks.add_task(log_request, "user_recommendations", user_id=user_id)
        
        # Получение рекомендаций от модели
        recommendations = manager.get_recommendations(
            user_id=user_id,
            model_name=model,
            num_recommendations=num_recommendations,
            exclude_seen=exclude_seen
        )
        
        # Форматирование ответа
        recommendation_items = []
        for rank, (item_id, score) in enumerate(recommendations, 1):
            item_info = manager.get_item_info(item_id)
            
            recommendation_items.append(RecommendationItem(
                item_id=item_id,
                score=float(score),
                rank=rank,
                item_name=item_info.get('name'),
                category=item_info.get('category')
            ))
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendation_items,
            model_used=model
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@app.get("/similar_items/{item_id}", response_model=SimilarItemsResponse)
async def get_similar_items(
    item_id: int = Path(..., gt=0, description="ID товара"),
    num_items: int = Query(10, ge=1, le=50, description="Количество похожих товаров"),
    model: str = Query("als", description="Модель для определения схожести"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    manager: ModelManager = Depends(get_model_manager)
):
    """Получить товары, похожие на заданный товар."""
    try:
        # Логирование запроса
        background_tasks.add_task(log_request, "similar_items", item_id=item_id)
        
        # Получение похожих товаров от модели
        similar_items = manager.get_similar_items(
            item_id=item_id,
            model_name=model,
            num_items=num_items
        )
        
        # Форматирование ответа
        similar_item_list = []
        for rank, (similar_item_id, score) in enumerate(similar_items, 1):
            item_info = manager.get_item_info(similar_item_id)
            
            similar_item_list.append(RecommendationItem(
                item_id=similar_item_id,
                score=float(score),
                rank=rank,
                item_name=item_info.get('name'),
                category=item_info.get('category')
            ))
        
        return SimilarItemsResponse(
            item_id=item_id,
            similar_items=similar_item_list,
            model_used=model
        )
        
    except Exception as e:
        logger.error(f"Error getting similar items for item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get similar items: {str(e)}")


@app.get("/popular_items", response_model=PopularItemsResponse)
async def get_popular_items(
    num_items: int = Query(10, ge=1, le=100, description="Количество популярных товаров"),
    category: Optional[str] = Query(None, description="Фильтр по категории"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    manager: ModelManager = Depends(get_model_manager)
):
    """Получить популярные товары, опционально отфильтрованные по категории."""
    try:
        # Логирование запроса
        background_tasks.add_task(log_request, "popular_items")
        
        # Получение популярных товаров от модели
        popular_items = manager.get_popular_items(
            num_items=num_items,
            category=category
        )
        
        # Форматирование ответа
        popular_item_list = []
        for rank, (item_id, score) in enumerate(popular_items, 1):
            item_info = manager.get_item_info(item_id)
            
            popular_item_list.append(RecommendationItem(
                item_id=item_id,
                score=float(score),
                rank=rank,
                item_name=item_info.get('name'),
                category=item_info.get('category')
            ))
        
        return PopularItemsResponse(
            popular_items=popular_item_list,
            category=category
        )
        
    except Exception as e:
        logger.error(f"Error getting popular items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get popular items: {str(e)}")


@app.post("/recommendations/batch", response_model=BatchRecommendationResponse)
async def get_batch_recommendations(
    request: BatchRecommendationRequest,
    model: str = Query("hybrid", description="Модель для создания рекомендаций"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    manager: ModelManager = Depends(get_model_manager)
):
    """Получить рекомендации для нескольких пользователей в пакетном режиме."""
    try:
        # Логирование запроса
        background_tasks.add_task(log_request, "batch_recommendations")
        
        batch_recommendations = {}
        
        for user_id in request.user_ids:
            try:
                recommendations = manager.get_recommendations(
                    user_id=user_id,
                    model_name=model,
                    num_recommendations=request.num_recommendations,
                    exclude_seen=request.exclude_seen
                )
                
                recommendation_items = []
                for rank, (item_id, score) in enumerate(recommendations, 1):
                    item_info = manager.get_item_info(item_id)
                    
                    recommendation_items.append(RecommendationItem(
                        item_id=item_id,
                        score=float(score),
                        rank=rank,
                        item_name=item_info.get('name'),
                        category=item_info.get('category')
                    ))
                
                batch_recommendations[user_id] = recommendation_items
                
            except Exception as e:
                logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                batch_recommendations[user_id] = []
        
        return BatchRecommendationResponse(
            recommendations=batch_recommendations,
            model_used=model,
            total_users=len(request.user_ids)
        )
        
    except Exception as e:
        logger.error(f"Error in batch recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get batch recommendations: {str(e)}")


@app.get("/models/status", response_model=ModelsStatusResponse)
async def get_models_status(manager: ModelManager = Depends(get_model_manager)):
    """Получить статус всех загруженных моделей."""
    try:
        status = manager.get_models_status()
        
        available_models = []
        for model_name, metadata in status['model_metadata'].items():
            available_models.append(ModelInfo(
                name=model_name,
                type=metadata.get('type', 'unknown'),
                version="1.0.0",
                trained_at=metadata.get('loaded_at')
            ))
        
        return ModelsStatusResponse(
            available_models=available_models,
            active_model="hybrid"
        )
        
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models status: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик исключений."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
