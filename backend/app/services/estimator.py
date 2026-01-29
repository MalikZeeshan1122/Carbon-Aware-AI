
# Coefficients based on academic research (e.g., Luccioni et al. 2023, Hugging Face)
# Wh per 1000 tokens (approximate) - Updated for 2026 models
MODEL_ENERGY_TABLE = {
    # OpenAI Models
    "gpt-5": 0.6,               # Next-gen flagship
    "gpt-4.5-turbo": 0.45,      # Enhanced GPT-4
    "gpt-4o": 0.4,              # Optimized GPT-4
    "gpt-4o-mini": 0.12,        # Smaller optimized
    "gpt-4-turbo": 0.35,        # Previous gen turbo
    "gpt-4": 0.4,               # Legacy support
    "gpt-3.5-turbo": 0.04,      # Legacy support
    # --- Aliases for frontend compatibility ---
    "gpt4": 0.4,                # Alias for gpt-4
    "gpt-4.0": 0.4,             # Alias for gpt-4
    "gpt-3.5": 0.04,            # Alias for gpt-3.5-turbo
    "gpt3.5-turbo": 0.04,       # Alias for gpt-3.5-turbo
    "gpt3.5": 0.04,             # Alias for gpt-3.5-turbo
    "gpt3": 0.04,               # Alias for gpt-3.5-turbo (if used)
    "o1": 0.5,                  # Reasoning model
    "o1-mini": 0.15,            # Smaller reasoning
    "o1-pro": 0.7,              # Pro reasoning
    "o3": 0.8,                  # Advanced reasoning
    "o3-mini": 0.25,            # Smaller o3
    
    # Anthropic Models
    "claude-4-opus": 0.7,       # Latest flagship
    "claude-4-sonnet": 0.3,     # Latest balanced
    "claude-3.5-opus": 0.6,     # Previous flagship
    "claude-3.5-sonnet": 0.15,  # Previous balanced
    "claude-3.5-haiku": 0.04,   # Previous efficient
    "claude-3-opus": 0.5,       # Legacy
    "claude-3-sonnet": 0.08,    # Legacy
    "claude-3-haiku": 0.02,     # Legacy
    
    # Google Models
    "gemini-2.0-ultra": 0.55,   # Latest flagship
    "gemini-2.0-pro": 0.3,      # Latest pro
    "gemini-2.0-flash": 0.08,   # Latest fast
    "gemini-1.5-ultra": 0.5,    # Previous flagship
    "gemini-1.5-pro": 0.15,     # Previous pro
    "gemini-1.5-flash": 0.06,   # Previous fast
    "gemini-pro": 0.06,         # Legacy support
    "gemini-ultra": 0.45,       # Legacy support
    
    # Meta Models
    "llama-4-405b": 0.35,       # Latest largest
    "llama-4-70b": 0.1,         # Latest large
    "llama-4-8b": 0.02,         # Latest small
    "llama-3.3-70b": 0.09,      # Previous
    "llama-3.2-90b-vision": 0.12, # Multimodal
    "llama-3.1-405b": 0.3,      # Previous large
    "llama-3-70b": 0.08,        # Legacy
    "llama-3-8b": 0.015,        # Legacy
    "llama-2-70b": 0.07,        # Legacy
    "llama-2-13b": 0.025,       # Legacy
    "llama-2-7b": 0.012,        # Legacy
    
    # Mistral Models
    "mistral-large-2": 0.15,    # Latest large
    "mistral-medium": 0.08,     # Medium
    "mistral-small": 0.03,      # Small
    "mixtral-8x22b": 0.12,      # MoE large
    "mixtral-8x7b": 0.05,       # MoE
    "codestral": 0.04,          # Code model
    "mistral-large": 0.1,       # Legacy
    
    # Cohere Models
    "command-r-plus": 0.1,      # Enhanced
    "command-r": 0.04,          # Standard
    "cohere-command": 0.04,     # Legacy
    "cohere-command-light": 0.015, # Legacy
    
    # xAI Models
    "grok-3": 0.5,              # Latest
    "grok-2": 0.15,             # Previous
    
    # Amazon Models
    "amazon-nova-pro": 0.12,    # Pro
    "amazon-nova-lite": 0.05,   # Lite
    "amazon-titan-express": 0.04, # Express
    
    # DeepSeek Models
    "deepseek-v3": 0.25,        # Latest
    "deepseek-coder-v2": 0.1,   # Code
    "deepseek-r1": 0.3,         # Reasoning
    
    # Alibaba Models
    "qwen-2.5-max": 0.15,       # Max
    "qwen-2.5-72b": 0.08,       # Large
    "qwen-2.5-coder-32b": 0.05, # Code
    
    # Microsoft Models
    "phi-4": 0.02,              # Latest efficient
    "phi-3-medium": 0.018,      # Previous medium
    "phi-3-mini": 0.008,        # Mini
    
    # Google Efficient Models
    "gemma-2-27b": 0.035,       # Medium
    "gemma-2-9b": 0.015,        # Small
    
    # Micro/Efficient Models
    "distilbert": 0.005,
    "bert-base": 0.008,
    "flan-t5-xxl": 0.025,
    "flan-t5-base": 0.008,
    "flan-t5-large": 0.02,
    "palm-2": 0.05,
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

    import sys
    print(f"[DEBUG] estimate_emissions called with model_name='{model_name}'", file=sys.stderr)

    model_key = model_name.lower()
    # Robust fallback: try direct, then alias, then default to gpt-3.5-turbo
    aliases = {
        "gpt-4": ["gpt4", "gpt-4.0", "gpt-4-turbo"],
        "gpt-3.5-turbo": ["gpt-3.5", "gpt3.5-turbo", "gpt3.5", "gpt3"]
    }
    energy_per_1k = None
    if model_key in MODEL_ENERGY_TABLE:
        energy_per_1k = MODEL_ENERGY_TABLE[model_key]
    else:
        # Try aliases
        for canonical, alist in aliases.items():
            if model_key == canonical or model_key in alist:
                energy_per_1k = MODEL_ENERGY_TABLE.get(canonical)
                break
    if energy_per_1k is None:
        # Fallback to gpt-3.5-turbo as a default
        energy_per_1k = MODEL_ENERGY_TABLE.get("gpt-3.5-turbo", 0.04)
        methodology_note = f"Model '{model_name}' not found. Used gpt-3.5-turbo as fallback."
        print(f"[DEBUG] Model '{model_name}' not found. Using fallback value {energy_per_1k}", file=sys.stderr)
    else:
        methodology_note = f"Model '{model_name}' resolved to '{model_key}'."
        print(f"[DEBUG] Model '{model_name}' resolved to '{model_key}' with value {energy_per_1k}", file=sys.stderr)

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
        "methodology": f"Coefficient-based: {energy_per_1k} Wh/1k tokens, PUE={DEFAULT_PUE}, CI={ci_factor} g/kWh. {methodology_note}"
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
