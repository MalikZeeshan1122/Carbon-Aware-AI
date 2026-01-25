from pydantic import BaseModel
from typing import Optional, List

class EstimationRequest(BaseModel):
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    region: Optional[str] = "global"

class EstimationResponse(BaseModel):
    energy_kwh: float
    co2_grams: float
    methodology: str

class RecommendationRequest(BaseModel):
    task_type: str  # e.g., "classification", "summarization", "reasoning"
    current_model: str

class RecommendationResponse(BaseModel):
    recommended_model: str
    savings_percentage: float
    reason: str

# --- NEW: Batch Estimation ---
class BatchEstimationRequest(BaseModel):
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    region: Optional[str] = "global"
    num_queries: int = 1

class BatchEstimationResponse(BaseModel):
    total_energy_kwh: float
    total_co2_grams: float
    per_query_energy_kwh: float
    per_query_co2_grams: float
    annual_projection_kg: float
    methodology: str

# --- NEW: Model Comparison ---
class ModelComparisonRequest(BaseModel):
    models: List[str]
    prompt_tokens: int
    completion_tokens: int
    region: Optional[str] = "global"

class ModelEmission(BaseModel):
    model: str
    energy_kwh: float
    co2_grams: float

class ModelComparisonResponse(BaseModel):
    comparisons: List[ModelEmission]
    most_efficient: str
    least_efficient: str

# --- NEW: Carbon Budget ---
class CarbonBudgetRequest(BaseModel):
    monthly_budget_kg: float
    model_name: str
    avg_tokens_per_query: int
    region: Optional[str] = "global"

class CarbonBudgetResponse(BaseModel):
    queries_allowed: int
    daily_limit: int
    budget_kg: float
    per_query_g: float
