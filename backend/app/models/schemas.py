from pydantic import BaseModel
from typing import Optional, List, Dict, Any

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


# ============================================================================
# ADVANCED FEATURE SCHEMAS
# ============================================================================

# --- Organization Management ---
class CreateOrganizationRequest(BaseModel):
    org_id: str
    name: str
    monthly_budget_kg: float

class AddUserRequest(BaseModel):
    user_id: str
    name: str
    role: Optional[str] = "member"

class LogUsageRequest(BaseModel):
    org_id: str
    user_id: str
    model_type: str  # text, image, video
    model_name: str
    co2_grams: float
    energy_kwh: float
    metadata: Optional[Dict[str, Any]] = None

# --- Carbon Offset ---
class OffsetCalculationRequest(BaseModel):
    co2_kg: float
    offset_type: Optional[str] = "tree_planting"
    time_horizon_years: Optional[int] = 1

# --- Alert Configuration ---
class AlertConfigRequest(BaseModel):
    budget_threshold_percent: Optional[int] = 80
    daily_limit_kg: Optional[float] = None
    model_restrictions: Optional[List[str]] = None
    email_notifications: Optional[bool] = True

# --- Training Estimation ---
class TrainingEstimationRequest(BaseModel):
    model_type: Optional[str] = "custom-medium"
    gpu_hours: Optional[int] = None
    gpu_type: Optional[str] = "a100"
    num_gpus: Optional[int] = 8
    region: Optional[str] = "global"
    parameters_billions: Optional[float] = None

# --- Report Generation ---
class ReportRequest(BaseModel):
    org_id: Optional[str] = None
    report_type: Optional[str] = "monthly"  # monthly, weekly, annual
    format: Optional[str] = "json"  # json, csv

# --- Scheduling ---
class ScheduleOptimizationRequest(BaseModel):
    region: str
    duration_hours: Optional[int] = 1

# --- Comparative Analytics ---
class ComparativeAnalyticsRequest(BaseModel):
    org_id: str
    industry: Optional[str] = "tech_startup"

# --- Forecasting ---
class ForecastRequest(BaseModel):
    org_id: Optional[str] = None
    days_ahead: Optional[int] = 30

