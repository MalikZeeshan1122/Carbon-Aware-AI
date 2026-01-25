
# Coefficients based on academic research (e.g., Luccioni et al. 2023, Hugging Face)
# Wh per 1000 tokens (approximate)
MODEL_ENERGY_TABLE = {
    "gpt-4": 0.4,       # High estimate for very large model
    "gpt-4-turbo": 0.35,
    "gpt-3.5-turbo": 0.04,
    "claude-3-opus": 0.5,
    "claude-3-sonnet": 0.08,
    "claude-3-haiku": 0.02,
    "mistral-large": 0.1,
    "mistral-medium": 0.05,
    "llama-3-70b": 0.08,
    "llama-3-8b": 0.015,
    "llama-2-70b": 0.07,
    "llama-2-13b": 0.025,
    "llama-2-7b": 0.012,
    "distilbert": 0.005,
    "bert-base": 0.008,
    "flan-t5-base": 0.008,
    "flan-t5-large": 0.02,
    "gemini-pro": 0.06,
    "gemini-ultra": 0.45,
    "palm-2": 0.05,
    "cohere-command": 0.04,
    "cohere-command-light": 0.015
}

# ============================================================================
# IMAGE GENERATION MODELS - Energy per image (Wh)
# Based on: Luccioni et al. (2023) "Power Hungry Processing: Watts Driving the Cost of AI Deployment"
# and Stable Diffusion benchmarks
# ============================================================================
IMAGE_GENERATION_ENERGY = {
    # Model: Wh per image at default resolution
    # Diffusion Models (high compute - many denoising steps)
    "dall-e-3": 2.9,           # ~2.9 Wh per 1024x1024 image (estimated from GPU benchmarks)
    "dall-e-2": 1.5,           # Older, less compute intensive
    "midjourney-v6": 3.2,      # High quality, estimated similar to DALL-E 3
    "midjourney-v5": 2.5,
    
    # Stable Diffusion variants
    "stable-diffusion-xl": 0.8,      # ~0.8 Wh on A100 (50 steps)
    "stable-diffusion-3": 1.2,       # More complex architecture
    "stable-diffusion-2.1": 0.5,     # ~0.5 Wh on A100 (50 steps)
    "stable-diffusion-1.5": 0.3,     # Lighter model
    
    # Fast/Efficient models
    "sdxl-turbo": 0.15,        # 1-4 step generation
    "lcm": 0.1,                # Latent Consistency Models - very fast
    "sdxl-lightning": 0.12,    # 4-step generation
    
    # Google/Meta
    "imagen": 3.5,             # Google's Imagen (estimated)
    "imagen-2": 4.0,           # Larger model
    "emu": 2.0,                # Meta's Emu
    
    # Other
    "flux-pro": 1.5,           # Black Forest Labs
    "flux-schnell": 0.4,       # Fast variant
    "kandinsky": 0.6,          # Open source
    "deepfloyd-if": 1.8,       # High quality, compute heavy
}

# Resolution multipliers for image generation
IMAGE_RESOLUTION_MULTIPLIER = {
    "256x256": 0.25,
    "512x512": 0.5,
    "768x768": 0.75,
    "1024x1024": 1.0,      # Base resolution for most benchmarks
    "1536x1536": 1.5,
    "2048x2048": 2.0,
    "4096x4096": 4.0,      # Very high res
}

# Step multipliers (default is 50 steps for SD models)
STEP_MULTIPLIER_BASE = 50

# ============================================================================
# VIDEO GENERATION MODELS - Energy per second of video (Wh)
# Based on: Estimates from GPU benchmarks, model architecture analysis
# Video generation is MUCH more compute intensive than images
# ============================================================================
VIDEO_GENERATION_ENERGY = {
    # Model: Wh per second of generated video at default resolution/fps
    # OpenAI
    "sora": 50.0,              # Estimated: Very high compute (transformer-based)
    "sora-turbo": 30.0,        # Hypothetical faster variant
    
    # Runway
    "runway-gen3": 25.0,       # ~25 Wh per second of video
    "runway-gen2": 15.0,       # Previous generation
    "runway-gen1": 8.0,
    
    # Google
    "veo": 35.0,               # Google's video model
    "lumiere": 40.0,           # High quality, long videos
    
    # Meta
    "make-a-video": 20.0,
    "emu-video": 18.0,
    
    # Pika Labs
    "pika-1.0": 12.0,
    "pika-1.5": 15.0,
    
    # Stability AI
    "stable-video-diffusion": 10.0,
    "stable-video-diffusion-xt": 15.0,
    
    # Open Source / Research
    "modelscope": 8.0,
    "zeroscope": 6.0,
    "animatediff": 5.0,
}

# Video resolution/quality multipliers
VIDEO_RESOLUTION_MULTIPLIER = {
    "360p": 0.25,
    "480p": 0.5,
    "720p": 0.75,
    "1080p": 1.0,      # Base resolution
    "2k": 1.5,
    "4k": 3.0,
}

# FPS multipliers (base is 24 fps)
VIDEO_FPS_MULTIPLIER = {
    12: 0.5,
    24: 1.0,      # Base
    30: 1.25,
    60: 2.0,
}

# Duration impact (non-linear - longer videos are more efficient per second)
def get_duration_efficiency(seconds: int) -> float:
    """Longer videos have some efficiency gains from batching."""
    if seconds <= 4:
        return 1.0
    elif seconds <= 10:
        return 0.9
    elif seconds <= 30:
        return 0.85
    else:
        return 0.8

# Carbon Intensity (gCO2/kWh) by Region (2023/2024 avg)
REGION_CI_TABLE = {
    "global": 475,      # World average
    "us-east": 350,     # Virginia, Ohio
    "us-west": 200,     # California/Oregon/Washington (Low)
    "us-central": 400,  # Texas, Iowa
    "eu-west": 250,     # Ireland, Netherlands
    "eu-north": 50,     # Sweden, Norway (Very Low - Hydro)
    "uk": 200,          # UK Grid
    "asiapac": 600,     # Singapore, Japan
    "india": 700,       # India (High Coal)
    "australia": 500,   # Australia
    "brazil": 100,      # Brazil (Hydro dominant)
    "canada": 120       # Canada (Hydro/Nuclear)
}

DEFAULT_PUE = 1.2  # Power Usage Effectiveness of Hyperscale Data Centers

def estimate_emissions(model_name: str, prompt_tokens: int, completion_tokens: int, region: str = "global") -> dict:
    """
    Estimate energy consumption and CO2 emissions for a single AI query.
    
    Formula:
    - Energy (Wh) = (Total_Tokens / 1000) × Energy_per_1k_tokens × PUE
    - CO2 (g) = Energy (kWh) × Carbon_Intensity (g/kWh)
    """
    # 1. Look up Energy per 1k tokens (fallback to GPT-3.5 level if unknown)
    energy_per_1k = MODEL_ENERGY_TABLE.get(model_name.lower(), 0.04) 
    
    total_tokens = prompt_tokens + completion_tokens
    
    # Energy (Wh) = (Tokens / 1000) * Energy_per_1k
    energy_wh = (total_tokens / 1000.0) * energy_per_1k
    
    # Adjust for Data Center Efficiency (PUE)
    energy_wh_total = energy_wh * DEFAULT_PUE
    
    # Convert to kWh
    energy_kwh = energy_wh_total / 1000.0
    
    # 2. Look up Carbon Intensity
    ci_factor = REGION_CI_TABLE.get(region.lower(), REGION_CI_TABLE["global"])
    
    # CO2 (grams) = Energy (kWh) * CI (g/kWh)
    co2_grams = energy_kwh * ci_factor
    
    return {
        "energy_kwh": energy_kwh,
        "co2_grams": co2_grams,
        "methodology": f"Coefficient-based: {energy_per_1k} Wh/1k tokens, PUE={DEFAULT_PUE}, CI={ci_factor} g/kWh"
    }

def estimate_batch(model_name: str, prompt_tokens: int, completion_tokens: int, region: str, num_queries: int) -> dict:
    """
    Estimate emissions for a batch of queries with annual projection.
    """
    single = estimate_emissions(model_name, prompt_tokens, completion_tokens, region)
    
    total_energy = single['energy_kwh'] * num_queries
    total_co2 = single['co2_grams'] * num_queries
    
    # Annual projection: assume this batch runs daily for a year
    annual_co2_kg = (total_co2 * 365) / 1000
    
    return {
        "total_energy_kwh": total_energy,
        "total_co2_grams": total_co2,
        "per_query_energy_kwh": single['energy_kwh'],
        "per_query_co2_grams": single['co2_grams'],
        "annual_projection_kg": annual_co2_kg,
        "methodology": single['methodology']
    }

def compare_models(models: list, prompt_tokens: int, completion_tokens: int, region: str) -> dict:
    """
    Compare emissions across multiple models for the same query.
    """
    comparisons = []
    
    for model in models:
        est = estimate_emissions(model, prompt_tokens, completion_tokens, region)
        comparisons.append({
            "model": model,
            "energy_kwh": est['energy_kwh'],
            "co2_grams": est['co2_grams']
        })
    
    # Sort by CO2
    comparisons.sort(key=lambda x: x['co2_grams'])
    
    return {
        "comparisons": comparisons,
        "most_efficient": comparisons[0]['model'] if comparisons else None,
        "least_efficient": comparisons[-1]['model'] if comparisons else None
    }

def calculate_carbon_budget(monthly_budget_kg: float, model_name: str, avg_tokens: int, region: str) -> dict:
    """
    Calculate how many queries fit within a carbon budget.
    """
    est = estimate_emissions(model_name, avg_tokens // 2, avg_tokens // 2, region)
    per_query_g = est['co2_grams']
    
    budget_grams = monthly_budget_kg * 1000
    queries_allowed = int(budget_grams / per_query_g) if per_query_g > 0 else 0
    daily_limit = queries_allowed // 30
    
    return {
        "queries_allowed": queries_allowed,
        "daily_limit": daily_limit,
        "budget_kg": monthly_budget_kg,
        "per_query_g": per_query_g
    }

def get_all_models():
    """Return list of all supported models."""
    return list(MODEL_ENERGY_TABLE.keys())

def get_all_regions():
    """Return list of all supported regions with their carbon intensity."""
    return {region: ci for region, ci in REGION_CI_TABLE.items()}

# ============================================================================
# IMAGE GENERATION ESTIMATION
# ============================================================================
def estimate_image_generation(
    model_name: str,
    num_images: int = 1,
    resolution: str = "1024x1024",
    steps: int = 50,
    region: str = "global"
) -> dict:
    """
    Estimate energy and CO2 for image generation.
    
    Args:
        model_name: Image generation model (dall-e-3, stable-diffusion-xl, etc.)
        num_images: Number of images to generate
        resolution: Image resolution (512x512, 1024x1024, etc.)
        steps: Number of denoising steps (for diffusion models)
        region: Cloud region for carbon intensity
    """
    # Get base energy per image
    base_energy_wh = IMAGE_GENERATION_ENERGY.get(model_name.lower(), 1.0)
    
    # Apply resolution multiplier
    res_mult = IMAGE_RESOLUTION_MULTIPLIER.get(resolution, 1.0)
    
    # Apply steps multiplier (relative to base 50 steps)
    step_mult = steps / STEP_MULTIPLIER_BASE
    
    # Calculate total energy per image
    energy_per_image_wh = base_energy_wh * res_mult * step_mult
    
    # Total energy for all images
    total_energy_wh = energy_per_image_wh * num_images
    
    # Apply PUE
    total_energy_wh *= DEFAULT_PUE
    
    # Convert to kWh
    energy_kwh = total_energy_wh / 1000.0
    
    # Get carbon intensity
    ci = REGION_CI_TABLE.get(region.lower(), REGION_CI_TABLE["global"])
    
    # Calculate CO2
    co2_grams = energy_kwh * ci
    
    # Context comparisons
    phone_charges = energy_kwh / 0.005  # ~5Wh per phone charge
    google_searches = energy_kwh / 0.0003  # ~0.3Wh per Google search
    
    return {
        "model": model_name,
        "num_images": num_images,
        "resolution": resolution,
        "steps": steps,
        "energy_per_image_wh": energy_per_image_wh / DEFAULT_PUE,  # Raw GPU energy
        "total_energy_kwh": energy_kwh,
        "co2_grams": co2_grams,
        "equivalent_phone_charges": phone_charges,
        "equivalent_google_searches": google_searches,
        "methodology": f"Base: {base_energy_wh} Wh/image, Res: {res_mult}x, Steps: {step_mult}x, PUE: {DEFAULT_PUE}, CI: {ci} g/kWh"
    }

# ============================================================================
# VIDEO GENERATION ESTIMATION
# ============================================================================
def estimate_video_generation(
    model_name: str,
    duration_seconds: int = 4,
    resolution: str = "1080p",
    fps: int = 24,
    region: str = "global"
) -> dict:
    """
    Estimate energy and CO2 for video generation.
    
    Args:
        model_name: Video generation model (sora, runway-gen3, etc.)
        duration_seconds: Length of video in seconds
        resolution: Video resolution (720p, 1080p, 4k)
        fps: Frames per second
        region: Cloud region for carbon intensity
    """
    # Get base energy per second of video
    base_energy_wh_per_sec = VIDEO_GENERATION_ENERGY.get(model_name.lower(), 15.0)
    
    # Apply resolution multiplier
    res_mult = VIDEO_RESOLUTION_MULTIPLIER.get(resolution, 1.0)
    
    # Apply FPS multiplier
    fps_mult = VIDEO_FPS_MULTIPLIER.get(fps, 1.0)
    
    # Apply duration efficiency (longer videos are slightly more efficient per second)
    duration_eff = get_duration_efficiency(duration_seconds)
    
    # Calculate energy per second
    energy_per_sec_wh = base_energy_wh_per_sec * res_mult * fps_mult * duration_eff
    
    # Total energy for full video
    total_energy_wh = energy_per_sec_wh * duration_seconds
    
    # Apply PUE
    total_energy_wh *= DEFAULT_PUE
    
    # Convert to kWh
    energy_kwh = total_energy_wh / 1000.0
    
    # Get carbon intensity
    ci = REGION_CI_TABLE.get(region.lower(), REGION_CI_TABLE["global"])
    
    # Calculate CO2
    co2_grams = energy_kwh * ci
    
    # Context comparisons
    phone_charges = energy_kwh / 0.005
    streaming_hours = co2_grams / 36  # ~36g CO2 per hour of Netflix streaming
    car_meters = co2_grams / 0.21  # ~210g CO2 per km, so 0.21g per meter
    
    return {
        "model": model_name,
        "duration_seconds": duration_seconds,
        "resolution": resolution,
        "fps": fps,
        "energy_per_second_wh": energy_per_sec_wh / DEFAULT_PUE,  # Raw GPU energy
        "total_energy_kwh": energy_kwh,
        "co2_grams": co2_grams,
        "equivalent_phone_charges": phone_charges,
        "equivalent_streaming_hours": streaming_hours,
        "equivalent_car_meters": car_meters,
        "methodology": f"Base: {base_energy_wh_per_sec} Wh/sec, Res: {res_mult}x, FPS: {fps_mult}x, Duration Eff: {duration_eff}, PUE: {DEFAULT_PUE}, CI: {ci} g/kWh"
    }

def get_image_models():
    """Return list of all supported image generation models."""
    return list(IMAGE_GENERATION_ENERGY.keys())

def get_video_models():
    """Return list of all supported video generation models."""
    return list(VIDEO_GENERATION_ENERGY.keys())

def compare_image_models(models: list, num_images: int, resolution: str, steps: int, region: str) -> dict:
    """Compare emissions across multiple image generation models."""
    comparisons = []
    for model in models:
        est = estimate_image_generation(model, num_images, resolution, steps, region)
        comparisons.append({
            "model": model,
            "energy_kwh": est['total_energy_kwh'],
            "co2_grams": est['co2_grams']
        })
    comparisons.sort(key=lambda x: x['co2_grams'])
    return {
        "comparisons": comparisons,
        "most_efficient": comparisons[0]['model'] if comparisons else None,
        "least_efficient": comparisons[-1]['model'] if comparisons else None
    }

def compare_video_models(models: list, duration: int, resolution: str, fps: int, region: str) -> dict:
    """Compare emissions across multiple video generation models."""
    comparisons = []
    for model in models:
        est = estimate_video_generation(model, duration, resolution, fps, region)
        comparisons.append({
            "model": model,
            "energy_kwh": est['total_energy_kwh'],
            "co2_grams": est['co2_grams']
        })
    comparisons.sort(key=lambda x: x['co2_grams'])
    return {
        "comparisons": comparisons,
        "most_efficient": comparisons[0]['model'] if comparisons else None,
        "least_efficient": comparisons[-1]['model'] if comparisons else None
    }
