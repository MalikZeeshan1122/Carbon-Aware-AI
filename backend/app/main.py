from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import (
    EstimationRequest, EstimationResponse,
    RecommendationRequest, RecommendationResponse,
    BatchEstimationRequest, BatchEstimationResponse,
    ModelComparisonRequest, ModelComparisonResponse,
    CarbonBudgetRequest, CarbonBudgetResponse,
    CreateOrganizationRequest, AddUserRequest, LogUsageRequest,
    OffsetCalculationRequest, AlertConfigRequest, TrainingEstimationRequest,
    ReportRequest, ScheduleOptimizationRequest, ComparativeAnalyticsRequest,
    ForecastRequest
)
from app.services.estimator import (
    estimate_emissions, estimate_batch, compare_models,
    calculate_carbon_budget, get_all_models, get_all_regions,
    estimate_image_generation, estimate_video_generation,
    get_image_models, get_video_models,
    compare_image_models, compare_video_models,
    IMAGE_GENERATION_ENERGY, VIDEO_GENERATION_ENERGY
)
from app.services.suggester import recommend_model
from app.services.datacenters import (
    get_all_datacenters, get_datacenter, get_datacenters_by_provider,
    get_cleanest_datacenters, get_datacenter_comparison
)
from app.services.advanced_features import (
    get_current_carbon_intensity, get_optimal_schedule, generate_24h_forecast,
    calculate_offset_requirements,
    create_organization, add_user_to_org, log_usage, get_org_dashboard,
    calculate_sustainability_score, estimate_training_emissions,
    generate_report, configure_alerts, check_alerts,
    get_api_analytics, log_api_call, forecast_emissions,
    get_comparative_analytics, ORGANIZATIONS, USERS
)

app = FastAPI(
    title="Carbon-Aware AI API",
    description="Comprehensive API to estimate, track, and reduce the carbon footprint of AI operations.",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Carbon-Aware AI API v1.0.0",
        "description": "Complete carbon footprint estimation and management for AI operations",
        "endpoints": {
            "core": ["/estimate", "/recommend", "/batch", "/compare", "/budget"],
            "models": ["/models", "/regions", "/image/models", "/video/models"],
            "datacenters": ["/datacenters", "/datacenters/comparison", "/datacenters/ranking/cleanest"],
            "real_time": ["/carbon/live", "/carbon/forecast", "/carbon/schedule"],
            "organization": ["/org/create", "/org/{org_id}/users", "/org/{org_id}/dashboard"],
            "analytics": ["/analytics/sustainability", "/analytics/comparative", "/analytics/api"],
            "advanced": ["/offset/calculate", "/training/estimate", "/report/generate", "/forecast"]
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# --- Core Endpoints ---

@app.post("/estimate", response_model=EstimationResponse)
def estimate_endpoint(request: EstimationRequest):
    """Estimate energy and CO2 for a single query."""
    result = estimate_emissions(
        model_name=request.model_name,
        prompt_tokens=request.prompt_tokens,
        completion_tokens=request.completion_tokens,
        region=request.region
    )
    return result

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_endpoint(request: RecommendationRequest):
    """Get right-sizing recommendation based on task type."""
    result = recommend_model(
        task_type=request.task_type,
        current_model=request.current_model
    )
    return result

# --- Advanced Endpoints ---

@app.post("/batch", response_model=BatchEstimationResponse)
def batch_estimate_endpoint(request: BatchEstimationRequest):
    """Estimate emissions for a batch of queries with annual projection."""
    result = estimate_batch(
        model_name=request.model_name,
        prompt_tokens=request.prompt_tokens,
        completion_tokens=request.completion_tokens,
        region=request.region,
        num_queries=request.num_queries
    )
    return result

@app.post("/compare", response_model=ModelComparisonResponse)
def compare_models_endpoint(request: ModelComparisonRequest):
    """Compare emissions across multiple models."""
    result = compare_models(
        models=request.models,
        prompt_tokens=request.prompt_tokens,
        completion_tokens=request.completion_tokens,
        region=request.region
    )
    return result

@app.post("/budget", response_model=CarbonBudgetResponse)
def carbon_budget_endpoint(request: CarbonBudgetRequest):
    """Calculate how many queries fit within a carbon budget."""
    result = calculate_carbon_budget(
        monthly_budget_kg=request.monthly_budget_kg,
        model_name=request.model_name,
        avg_tokens=request.avg_tokens_per_query,
        region=request.region
    )
    return result

# --- Utility Endpoints ---

@app.get("/models")
def list_models():
    """List all supported AI models."""
    return {"models": get_all_models()}

@app.get("/regions")
def list_regions():
    """List all supported regions with their carbon intensity."""
    return {"regions": get_all_regions()}

# --- Data Center Endpoints ---

@app.get("/datacenters")
def list_datacenters():
    """List all data centers with their carbon emission details."""
    return {"datacenters": get_all_datacenters()}

@app.get("/datacenters/comparison")
def get_comparison():
    """Get comparison data for all data centers (for visualization)."""
    return {"comparison": get_datacenter_comparison()}

@app.get("/datacenters/ranking/cleanest")
def get_cleanest():
    """Get the top 5 cleanest data centers by carbon intensity."""
    return {"cleanest": get_cleanest_datacenters(5)}

@app.get("/datacenters/provider/{provider}")
def get_provider_datacenters(provider: str):
    """Get all data centers for a specific provider (google, aws, microsoft)."""
    return {"datacenters": get_datacenters_by_provider(provider)}

@app.get("/datacenters/{dc_id}")
def get_datacenter_info(dc_id: str):
    """Get detailed information about a specific data center."""
    dc = get_datacenter(dc_id)
    if dc:
        return dc
    return {"error": "Data center not found"}

# ============================================================================
# IMAGE GENERATION ENDPOINTS
# ============================================================================

@app.get("/image/models")
def list_image_models():
    """List all supported image generation models with their base energy."""
    return {
        "models": get_image_models(),
        "energy_data": {k: {"wh_per_image": v} for k, v in IMAGE_GENERATION_ENERGY.items()}
    }

@app.get("/image/estimate")
def estimate_image(
    model_name: str = "stable-diffusion-xl",
    num_images: int = 1,
    resolution: str = "1024x1024",
    steps: int = 50,
    region: str = "global"
):
    """Estimate CO2 emissions for image generation."""
    return estimate_image_generation(model_name, num_images, resolution, steps, region)

@app.get("/image/compare")
def compare_images(
    models: str = "dall-e-3,stable-diffusion-xl,sdxl-turbo",
    num_images: int = 1,
    resolution: str = "1024x1024",
    steps: int = 50,
    region: str = "global"
):
    """Compare emissions across multiple image generation models."""
    model_list = [m.strip() for m in models.split(",")]
    return compare_image_models(model_list, num_images, resolution, steps, region)

# ============================================================================
# VIDEO GENERATION ENDPOINTS
# ============================================================================

@app.get("/video/models")
def list_video_models():
    """List all supported video generation models with their base energy."""
    return {
        "models": get_video_models(),
        "energy_data": {k: {"wh_per_second": v} for k, v in VIDEO_GENERATION_ENERGY.items()}
    }

@app.get("/video/estimate")
def estimate_video(
    model_name: str = "runway-gen3",
    duration_seconds: int = 4,
    resolution: str = "1080p",
    fps: int = 24,
    region: str = "global"
):
    """Estimate CO2 emissions for video generation."""
    return estimate_video_generation(model_name, duration_seconds, resolution, fps, region)

@app.get("/video/compare")
def compare_videos(
    models: str = "sora,runway-gen3,stable-video-diffusion",
    duration_seconds: int = 4,
    resolution: str = "1080p",
    fps: int = 24,
    region: str = "global"
):
    """Compare emissions across multiple video generation models."""
    model_list = [m.strip() for m in models.split(",")]
    return compare_video_models(model_list, duration_seconds, resolution, fps, region)


# ============================================================================
# REAL-TIME CARBON INTENSITY ENDPOINTS
# ============================================================================

@app.get("/carbon/live")
def get_live_carbon_intensity(region: str = "global"):
    """Get real-time carbon intensity for a region with 24h forecast."""
    return get_current_carbon_intensity(region)

@app.get("/carbon/forecast/{region}")
def get_carbon_forecast(region: str):
    """Get 24-hour carbon intensity forecast for a region."""
    return {
        "region": region,
        "forecast": generate_24h_forecast(region)
    }

@app.get("/carbon/schedule")
def get_optimal_run_time(region: str = "global", duration_hours: int = 1):
    """Find the optimal time window in the next 24h for lowest carbon emissions."""
    return get_optimal_schedule(region, duration_hours)

@app.post("/carbon/schedule")
def post_optimal_run_time(request: ScheduleOptimizationRequest):
    """Find the optimal time window in the next 24h for lowest carbon emissions."""
    return get_optimal_schedule(request.region, request.duration_hours)


# ============================================================================
# CARBON OFFSET ENDPOINTS
# ============================================================================

@app.get("/offset/calculate")
def calculate_offset(co2_kg: float, offset_type: str = "tree_planting"):
    """Calculate offset requirements and costs for given CO2 emissions."""
    return calculate_offset_requirements(co2_kg, offset_type)

@app.post("/offset/calculate")
def post_calculate_offset(request: OffsetCalculationRequest):
    """Calculate offset requirements and costs for given CO2 emissions."""
    return calculate_offset_requirements(
        request.co2_kg, 
        request.offset_type,
        request.time_horizon_years
    )


# ============================================================================
# ORGANIZATION & TEAM MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/org/create")
def create_org(request: CreateOrganizationRequest):
    """Create a new organization with a carbon budget."""
    return create_organization(request.org_id, request.name, request.monthly_budget_kg)

@app.get("/org/{org_id}")
def get_org(org_id: str):
    """Get organization details."""
    if org_id in ORGANIZATIONS:
        return ORGANIZATIONS[org_id]
    return {"error": "Organization not found"}

@app.get("/org/{org_id}/dashboard")
def get_dashboard(org_id: str):
    """Get comprehensive organization dashboard with usage stats."""
    return get_org_dashboard(org_id)

@app.post("/org/{org_id}/users")
def add_user(org_id: str, request: AddUserRequest):
    """Add a user to an organization."""
    return add_user_to_org(org_id, request.user_id, request.name, request.role)

@app.get("/org/{org_id}/users")
def list_org_users(org_id: str):
    """List all users in an organization."""
    if org_id not in ORGANIZATIONS:
        return {"error": "Organization not found"}
    user_ids = ORGANIZATIONS[org_id]["users"]
    return {"users": [USERS.get(uid) for uid in user_ids if uid in USERS]}

@app.post("/org/{org_id}/log")
def log_org_usage(org_id: str, request: LogUsageRequest):
    """Log carbon usage for an organization."""
    return log_usage(
        org_id=request.org_id,
        user_id=request.user_id,
        model_type=request.model_type,
        model_name=request.model_name,
        co2_grams=request.co2_grams,
        energy_kwh=request.energy_kwh,
        metadata=request.metadata
    )

@app.post("/org/{org_id}/alerts/configure")
def configure_org_alerts(org_id: str, request: AlertConfigRequest):
    """Configure alert settings for an organization."""
    return configure_alerts(
        org_id,
        request.budget_threshold_percent,
        request.daily_limit_kg,
        request.model_restrictions,
        request.email_notifications
    )

@app.get("/org/{org_id}/alerts/check")
def check_org_alerts(org_id: str, model_name: str = None):
    """Check if any alerts should be triggered for an organization."""
    return {"alerts": check_alerts(org_id, model_name)}


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/analytics/sustainability")
def get_sustainability_score(org_id: str = None):
    """Calculate sustainability score based on AI usage patterns."""
    return calculate_sustainability_score(org_id)

@app.get("/analytics/comparative")
def get_comparative(org_id: str, industry: str = "tech_startup"):
    """Compare organization's usage against industry benchmarks."""
    return get_comparative_analytics(org_id, industry)

@app.post("/analytics/comparative")
def post_comparative(request: ComparativeAnalyticsRequest):
    """Compare organization's usage against industry benchmarks."""
    return get_comparative_analytics(request.org_id, request.industry)

@app.get("/analytics/api")
def get_api_usage_analytics(hours: int = 24):
    """Get API usage analytics for the specified time period."""
    return get_api_analytics(hours)


# ============================================================================
# TRAINING EMISSIONS ENDPOINTS
# ============================================================================

@app.get("/training/estimate")
def estimate_training(
    model_type: str = "custom-medium",
    gpu_hours: int = None,
    gpu_type: str = "a100",
    num_gpus: int = 8,
    region: str = "global"
):
    """Estimate CO2 emissions for AI model training."""
    return estimate_training_emissions(
        model_type, gpu_hours, gpu_type, num_gpus, region
    )

@app.post("/training/estimate")
def post_estimate_training(request: TrainingEstimationRequest):
    """Estimate CO2 emissions for AI model training."""
    return estimate_training_emissions(
        request.model_type,
        request.gpu_hours,
        request.gpu_type,
        request.num_gpus,
        request.region,
        request.parameters_billions
    )


# ============================================================================
# REPORTING ENDPOINTS
# ============================================================================

@app.get("/report/generate")
def generate_emissions_report(
    org_id: str = None,
    report_type: str = "monthly",
    format: str = "json"
):
    """Generate a comprehensive carbon emissions report."""
    return generate_report(org_id, report_type, format)

@app.post("/report/generate")
def post_generate_report(request: ReportRequest):
    """Generate a comprehensive carbon emissions report."""
    return generate_report(request.org_id, request.report_type, request.format)


# ============================================================================
# FORECASTING ENDPOINTS
# ============================================================================

@app.get("/forecast/emissions")
def get_emissions_forecast(org_id: str = None, days_ahead: int = 30):
    """Forecast future emissions based on historical trends."""
    return forecast_emissions(org_id, days_ahead)

@app.post("/forecast/emissions")
def post_emissions_forecast(request: ForecastRequest):
    """Forecast future emissions based on historical trends."""
    return forecast_emissions(request.org_id, request.days_ahead)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
