"""
API schemas for the recommendation service.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class RecommendationRequest(BaseModel):
    """Request schema for user recommendations."""
    user_id: int = Field(..., gt=0, description="User ID")
    num_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations")
    exclude_seen: bool = Field(True, description="Exclude items user has already interacted with")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 12345,
                "num_recommendations": 10,
                "exclude_seen": True
            }
        }


class SimilarItemsRequest(BaseModel):
    """Request schema for similar items."""
    item_id: int = Field(..., gt=0, description="Item ID")
    num_items: int = Field(10, ge=1, le=50, description="Number of similar items")
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_id": 67890,
                "num_items": 10
            }
        }


class PopularItemsRequest(BaseModel):
    """Request schema for popular items."""
    num_items: int = Field(10, ge=1, le=100, description="Number of popular items")
    category: Optional[str] = Field(None, description="Filter by category")
    
    class Config:
        json_schema_extra = {
            "example": {
                "num_items": 20,
                "category": "Electronics"
            }
        }


class RecommendationItem(BaseModel):
    """Single recommendation item."""
    item_id: int = Field(..., description="Item ID")
    score: float = Field(..., description="Recommendation score")
    rank: int = Field(..., description="Rank in recommendation list")
    item_name: Optional[str] = Field(None, description="Item name")
    category: Optional[str] = Field(None, description="Item category")
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_id": 67890,
                "score": 0.85,
                "rank": 1,
                "item_name": "Wireless Headphones",
                "category": "Electronics"
            }
        }


class RecommendationResponse(BaseModel):
    """Response schema for recommendations."""
    user_id: int = Field(..., description="User ID")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommendations")
    model_used: str = Field(..., description="Model used for recommendations")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "user_id": 12345,
                "recommendations": [
                    {
                        "item_id": 67890,
                        "score": 0.85,
                        "rank": 1,
                        "item_name": "Wireless Headphones",
                        "category": "Electronics"
                    }
                ],
                "model_used": "hybrid",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class SimilarItemsResponse(BaseModel):
    """Response schema for similar items."""
    item_id: int = Field(..., description="Source item ID")
    similar_items: List[RecommendationItem] = Field(..., description="List of similar items")
    model_used: str = Field(..., description="Model used for similarity")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        protected_namespaces = ()


class PopularItemsResponse(BaseModel):
    """Response schema for popular items."""
    popular_items: List[RecommendationItem] = Field(..., description="List of popular items")
    category: Optional[str] = Field(None, description="Category filter applied")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": {
                    "collaborative": True,
                    "content_based": True,
                    "hybrid": True
                },
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid user ID",
                "detail": "User ID must be a positive integer",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class BatchRecommendationRequest(BaseModel):
    """Request schema for batch recommendations."""
    user_ids: List[int] = Field(..., min_items=1, max_items=1000, description="List of user IDs")
    num_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations per user")
    exclude_seen: bool = Field(True, description="Exclude items users have already interacted with")
    
    @validator('user_ids')
    def validate_user_ids(cls, v):
        if not all(uid > 0 for uid in v):
            raise ValueError('All user IDs must be positive integers')
        return v


class BatchRecommendationResponse(BaseModel):
    """Response schema for batch recommendations."""
    recommendations: Dict[int, List[RecommendationItem]] = Field(..., description="Recommendations per user")
    model_used: str = Field(..., description="Model used for recommendations")
    total_users: int = Field(..., description="Total number of users processed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        protected_namespaces = ()


class ModelInfo(BaseModel):
    """Model information schema."""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    trained_at: Optional[datetime] = Field(None, description="Training timestamp")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")


class ModelsStatusResponse(BaseModel):
    """Models status response."""
    available_models: List[ModelInfo] = Field(..., description="List of available models")
    active_model: str = Field(..., description="Currently active model")
    timestamp: datetime = Field(default_factory=datetime.now, description="Status timestamp")
