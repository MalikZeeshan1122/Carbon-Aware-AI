from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import (
    EstimationRequest, EstimationResponse,
    RecommendationRequest, RecommendationResponse,
    BatchEstimationRequest, BatchEstimationResponse,
    ModelComparisonRequest, ModelComparisonResponse,
    CarbonBudgetRequest, CarbonBudgetResponse
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

app = FastAPI(
    title="Carbon-Aware AI API",
    description="Estimate CO2 emissions and get right-sizing recommendations for AI models.",
    version="0.2.0"
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
        "message": "Welcome to Carbon-Aware AI API v0.2.0",
        "endpoints": ["/estimate", "/recommend", "/batch", "/compare", "/budget", "/models", "/regions"]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
