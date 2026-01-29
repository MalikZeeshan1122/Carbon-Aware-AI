# ============================================================================
# ADVANCED FEATURES MODULE
# Carbon-Aware AI - Complete Enterprise Features
# ============================================================================

from datetime import datetime, timedelta
from typing import Optional, List, Dict
import math
import random
import json

# ============================================================================
# 1. REAL-TIME CARBON INTENSITY SIMULATION
# Simulates live grid carbon intensity data (would connect to real APIs in production)
# ============================================================================

# Hourly carbon intensity patterns by region (gCO2/kWh)
# Based on typical daily patterns - solar peaks midday, wind varies
HOURLY_CI_PATTERNS = {
    "us-west": {
        "base": 150,
        "solar_reduction": 80,  # Solar peak reduction (11am-3pm)
        "peak_hours": [17, 18, 19, 20],  # Evening peak
        "peak_increase": 50
    },
    "us-east": {
        "base": 380,
        "solar_reduction": 60,
        "peak_hours": [17, 18, 19, 20, 21],
        "peak_increase": 80
    },
    "eu-west": {
        "base": 250,
        "solar_reduction": 70,
        "peak_hours": [17, 18, 19],
        "peak_increase": 60
    },
    "eu-north": {
        "base": 30,
        "solar_reduction": 10,
        "peak_hours": [17, 18, 19],
        "peak_increase": 20
    },
    "global": {
        "base": 475,
        "solar_reduction": 50,
        "peak_hours": [17, 18, 19, 20],
        "peak_increase": 70
    }
}

def get_current_carbon_intensity(region: str = "global") -> dict:
    """Get simulated real-time carbon intensity for a region."""
    now = datetime.now()
    hour = now.hour
    
    pattern = HOURLY_CI_PATTERNS.get(region.lower(), HOURLY_CI_PATTERNS["global"])
    base_ci = pattern["base"]
    
    # Apply solar reduction (11am - 3pm)
    if 11 <= hour <= 15:
        solar_factor = pattern["solar_reduction"] * math.sin((hour - 11) * math.pi / 4)
        base_ci -= solar_factor
    
    # Apply peak hours increase
    if hour in pattern["peak_hours"]:
        base_ci += pattern["peak_increase"]
    
    # Add some randomness for realism
    base_ci += random.uniform(-20, 20)
    
    # Determine status
    if base_ci < 100:
        status = "excellent"
        recommendation = "Great time for AI workloads!"
    elif base_ci < 250:
        status = "good"
        recommendation = "Good conditions for AI workloads"
    elif base_ci < 400:
        status = "moderate"
        recommendation = "Consider deferring non-urgent workloads"
    else:
        status = "poor"
        recommendation = "High carbon intensity - defer if possible"
    
    return {
        "region": region,
        "carbon_intensity": round(base_ci, 1),
        "unit": "gCO2/kWh",
        "timestamp": now.isoformat(),
        "status": status,
        "recommendation": recommendation,
        "forecast_24h": generate_24h_forecast(region)
    }

def generate_24h_forecast(region: str) -> list:
    """Generate 24-hour carbon intensity forecast."""
    now = datetime.now()
    pattern = HOURLY_CI_PATTERNS.get(region.lower(), HOURLY_CI_PATTERNS["global"])
    
    forecast = []
    for i in range(24):
        future_hour = (now.hour + i) % 24
        future_time = now + timedelta(hours=i)
        
        ci = pattern["base"]
        if 11 <= future_hour <= 15:
            ci -= pattern["solar_reduction"] * 0.7
        if future_hour in pattern["peak_hours"]:
            ci += pattern["peak_increase"]
        
        ci += random.uniform(-10, 10)
        
        forecast.append({
            "hour": future_time.strftime("%H:00"),
            "carbon_intensity": round(ci, 1),
            "relative": "low" if ci < 200 else "medium" if ci < 400 else "high"
        })
    
    return forecast

def get_optimal_schedule(region: str, duration_hours: int = 1) -> dict:
    """Find the optimal time window in the next 24h for lowest carbon."""
    forecast = generate_24h_forecast(region)
    
    # Find best window
    best_start = 0
    best_avg_ci = float('inf')
    
    for i in range(len(forecast) - duration_hours + 1):
        window = forecast[i:i + duration_hours]
        avg_ci = sum(h["carbon_intensity"] for h in window) / len(window)
        if avg_ci < best_avg_ci:
            best_avg_ci = avg_ci
            best_start = i
    
    # Find worst window for comparison
    worst_start = 0
    worst_avg_ci = 0
    for i in range(len(forecast) - duration_hours + 1):
        window = forecast[i:i + duration_hours]
        avg_ci = sum(h["carbon_intensity"] for h in window) / len(window)
        if avg_ci > worst_avg_ci:
            worst_avg_ci = avg_ci
            worst_start = i
    
    savings_percent = ((worst_avg_ci - best_avg_ci) / worst_avg_ci) * 100 if worst_avg_ci > 0 else 0
    
    return {
        "region": region,
        "optimal_start_time": forecast[best_start]["hour"],
        "optimal_end_time": forecast[min(best_start + duration_hours, 23)]["hour"],
        "optimal_avg_ci": round(best_avg_ci, 1),
        "worst_start_time": forecast[worst_start]["hour"],
        "worst_avg_ci": round(worst_avg_ci, 1),
        "potential_savings_percent": round(savings_percent, 1),
        "recommendation": f"Schedule workload at {forecast[best_start]['hour']} to save {savings_percent:.0f}% carbon"
    }


# ============================================================================
# 2. CARBON OFFSET CALCULATOR
# Calculate offset requirements and costs
# ============================================================================

# Offset prices per tonne CO2 (USD)
OFFSET_PRICES = {
    "tree_planting": 15,                # Forestry projects (nature-based, biodiversity)
    "renewable_energy": 12,             # Wind/Solar credits (most affordable)
    "methane_capture": 20,              # Landfill gas capture
    "direct_air_capture": 300,          # DAC (most permanent)
    "verified_carbon_standard": 25,     # VCS certified (high quality)
    "gold_standard": 35                 # Gold Standard certified (premium nature-based)
}

# Trees absorb ~21kg CO2 per year on average
TREE_CO2_ABSORPTION_KG_YEAR = 21

def calculate_offset_requirements(
    co2_kg: float,
    offset_type: str = "tree_planting",
    time_horizon_years: int = 1
) -> dict:
    """Calculate offset requirements for given CO2 emissions."""
    
    # Calculate trees needed
    trees_needed = math.ceil(co2_kg / TREE_CO2_ABSORPTION_KG_YEAR)
    trees_for_lifetime = math.ceil(co2_kg / (TREE_CO2_ABSORPTION_KG_YEAR * 40))  # 40 year tree lifetime
    
    # Calculate cost
    co2_tonnes = co2_kg / 1000
    price_per_tonne = OFFSET_PRICES.get(offset_type, 20)
    offset_cost = co2_tonnes * price_per_tonne
    
    # Real-world equivalents
    equivalents = {
        "car_km_equivalent": co2_kg / 0.21,  # 210g CO2/km
        "flights_nyc_london": co2_kg / 900,   # ~900kg per flight
        "years_of_streaming": co2_kg / 36,    # 36g per hour * 8760 hours
        "smartphones_charged": co2_kg * 1000 / 8.22,  # 8.22g per charge
        "beef_meals": co2_kg / 6.61,          # 6.61 kg per kg beef
        "gallons_gasoline": co2_kg / 8.89     # 8.89 kg per gallon
    }
    
    return {
        "co2_to_offset_kg": co2_kg,
        "co2_to_offset_tonnes": co2_tonnes,
        "offset_type": offset_type,
        "trees_needed_annual": trees_needed,
        "trees_needed_permanent": trees_for_lifetime,
        "cost_usd": round(offset_cost, 2),
        "price_per_tonne": price_per_tonne,
        "time_horizon_years": time_horizon_years,
        "real_world_equivalents": equivalents,
        "offset_options": {
            "tree_planting": {
                "price_per_tonne": OFFSET_PRICES["tree_planting"],
                "total_cost": round(co2_tonnes * OFFSET_PRICES["tree_planting"], 2),
                "description": "Nature-based, supports biodiversity, visible impact."
            },
            "renewable_energy": {
                "price_per_tonne": OFFSET_PRICES["renewable_energy"],
                "total_cost": round(co2_tonnes * OFFSET_PRICES["renewable_energy"], 2),
                "description": "Wind/solar projects, most affordable, grid decarbonization."
            },
            "methane_capture": {
                "price_per_tonne": OFFSET_PRICES["methane_capture"],
                "total_cost": round(co2_tonnes * OFFSET_PRICES["methane_capture"], 2),
                "description": "Landfill gas capture, high impact, prevents potent GHG."
            },
            "direct_air_capture": {
                "price_per_tonne": OFFSET_PRICES["direct_air_capture"],
                "total_cost": round(co2_tonnes * OFFSET_PRICES["direct_air_capture"], 2),
                "description": "Removes CO₂ permanently from atmosphere, most permanent."
            },
            "verified_carbon_standard": {
                "price_per_tonne": OFFSET_PRICES["verified_carbon_standard"],
                "total_cost": round(co2_tonnes * OFFSET_PRICES["verified_carbon_standard"], 2),
                "description": "VCS certified, high quality, third-party verified."
            },
            "gold_standard": {
                "price_per_tonne": OFFSET_PRICES["gold_standard"],
                "total_cost": round(co2_tonnes * OFFSET_PRICES["gold_standard"], 2),
                "description": "Gold Standard certified, premium nature-based, co-benefits."
            }
        }
    }


# ============================================================================
# 3. ORGANIZATION/TEAM CARBON TRACKING
# Multi-user carbon budget management
# ============================================================================

# In-memory storage (would be database in production)
ORGANIZATIONS = {}
USERS = {}
USAGE_LOG = []

def create_organization(org_id: str, name: str, monthly_budget_kg: float) -> dict:
    """Create a new organization with carbon budget."""
    ORGANIZATIONS[org_id] = {
        "id": org_id,
        "name": name,
        "monthly_budget_kg": monthly_budget_kg,
        "created_at": datetime.now().isoformat(),
        "users": [],
        "current_month_usage_kg": 0,
        "total_usage_kg": 0,
        "alerts_enabled": True,
        "alert_threshold_percent": 80
    }
    return ORGANIZATIONS[org_id]

def add_user_to_org(org_id: str, user_id: str, name: str, role: str = "member") -> dict:
    """Add a user to an organization."""
    if org_id not in ORGANIZATIONS:
        return {"error": "Organization not found"}
    
    user = {
        "id": user_id,
        "name": name,
        "org_id": org_id,
        "role": role,  # admin, member, viewer
        "personal_usage_kg": 0,
        "joined_at": datetime.now().isoformat()
    }
    USERS[user_id] = user
    ORGANIZATIONS[org_id]["users"].append(user_id)
    return user

def log_usage(
    org_id: str,
    user_id: str,
    model_type: str,  # text, image, video
    model_name: str,
    co2_grams: float,
    energy_kwh: float,
    metadata: dict = None
) -> dict:
    """Log carbon usage for tracking."""
    if org_id not in ORGANIZATIONS:
        return {"error": "Organization not found"}
    
    co2_kg = co2_grams / 1000
    
    log_entry = {
        "id": len(USAGE_LOG) + 1,
        "org_id": org_id,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "model_name": model_name,
        "co2_kg": co2_kg,
        "energy_kwh": energy_kwh,
        "metadata": metadata or {}
    }
    USAGE_LOG.append(log_entry)
    
    # Update org totals
    ORGANIZATIONS[org_id]["current_month_usage_kg"] += co2_kg
    ORGANIZATIONS[org_id]["total_usage_kg"] += co2_kg
    
    # Update user totals
    if user_id in USERS:
        USERS[user_id]["personal_usage_kg"] += co2_kg
    
    # Check alerts
    alert = None
    org = ORGANIZATIONS[org_id]
    usage_percent = (org["current_month_usage_kg"] / org["monthly_budget_kg"]) * 100
    
    if org["alerts_enabled"] and usage_percent >= org["alert_threshold_percent"]:
        alert = {
            "type": "budget_warning",
            "message": f"Carbon budget at {usage_percent:.1f}% ({org['current_month_usage_kg']:.2f} / {org['monthly_budget_kg']} kg)",
            "severity": "warning" if usage_percent < 100 else "critical"
        }
    
    return {
        "logged": True,
        "entry_id": log_entry["id"],
        "current_usage_kg": org["current_month_usage_kg"],
        "budget_remaining_kg": org["monthly_budget_kg"] - org["current_month_usage_kg"],
        "usage_percent": usage_percent,
        "alert": alert
    }

def get_org_dashboard(org_id: str) -> dict:
    """Get organization dashboard with usage stats."""
    if org_id not in ORGANIZATIONS:
        return {"error": "Organization not found"}
    
    org = ORGANIZATIONS[org_id]
    org_logs = [l for l in USAGE_LOG if l["org_id"] == org_id]
    
    # Calculate stats
    usage_by_model_type = {}
    usage_by_user = {}
    usage_by_model = {}
    daily_usage = {}
    
    for log in org_logs:
        # By model type
        mt = log["model_type"]
        usage_by_model_type[mt] = usage_by_model_type.get(mt, 0) + log["co2_kg"]
        
        # By user
        uid = log["user_id"]
        usage_by_user[uid] = usage_by_user.get(uid, 0) + log["co2_kg"]
        
        # By model
        mn = log["model_name"]
        usage_by_model[mn] = usage_by_model.get(mn, 0) + log["co2_kg"]
        
        # Daily
        day = log["timestamp"][:10]
        daily_usage[day] = daily_usage.get(day, 0) + log["co2_kg"]
    
    budget_used_percent = (org["current_month_usage_kg"] / org["monthly_budget_kg"]) * 100
    
    return {
        "organization": {
            "id": org["id"],
            "name": org["name"],
            "monthly_budget_kg": org["monthly_budget_kg"],
            "current_usage_kg": org["current_month_usage_kg"],
            "budget_remaining_kg": org["monthly_budget_kg"] - org["current_month_usage_kg"],
            "budget_used_percent": round(budget_used_percent, 1),
            "total_usage_kg": org["total_usage_kg"],
            "user_count": len(org["users"])
        },
        "breakdown": {
            "by_model_type": usage_by_model_type,
            "by_user": usage_by_user,
            "by_model": dict(sorted(usage_by_model.items(), key=lambda x: x[1], reverse=True)[:10]),
            "daily_trend": daily_usage
        },
        "sustainability_score": calculate_sustainability_score(org_id),
        "recommendations": generate_org_recommendations(org_id)
    }


# ============================================================================
# 4. SUSTAINABILITY SCORING
# Rate AI practices on sustainability
# ============================================================================

def calculate_sustainability_score(org_id: str = None) -> dict:
    """Calculate a sustainability score (0-100) based on AI usage patterns."""
    
    if org_id and org_id in ORGANIZATIONS:
        org_logs = [l for l in USAGE_LOG if l["org_id"] == org_id]
    else:
        org_logs = USAGE_LOG
    
    if not org_logs:
        return {
            "score": 50,
            "grade": "C",
            "breakdown": {},
            "message": "Not enough data to calculate score"
        }
    
    # Scoring factors
    scores = {}
    
    # 1. Model efficiency (are they using right-sized models?)
    efficient_models = ["distilbert", "llama-3-8b", "gpt-3.5-turbo", "sdxl-turbo", "lcm", "animatediff"]
    frontier_models = ["gpt-4", "claude-3-opus", "sora", "dall-e-3"]
    
    model_names = [l["model_name"].lower() for l in org_logs]
    efficient_ratio = sum(1 for m in model_names if m in efficient_models) / len(model_names) if model_names else 0
    frontier_ratio = sum(1 for m in model_names if m in frontier_models) / len(model_names) if model_names else 0
    
    scores["model_efficiency"] = min(100, int((efficient_ratio * 50) + ((1 - frontier_ratio) * 50)))
    
    # 2. Energy per query (lower is better)
    avg_energy = sum(l["energy_kwh"] for l in org_logs) / len(org_logs) if org_logs else 0
    scores["energy_efficiency"] = max(0, min(100, int(100 - (avg_energy * 10000))))
    
    # 3. Budget compliance
    if org_id and org_id in ORGANIZATIONS:
        org = ORGANIZATIONS[org_id]
        usage_percent = (org["current_month_usage_kg"] / org["monthly_budget_kg"]) * 100
        scores["budget_compliance"] = max(0, min(100, int(100 - max(0, usage_percent - 80))))
    else:
        scores["budget_compliance"] = 70
    
    # Calculate overall score
    weights = {
        "model_efficiency": 0.4,
        "energy_efficiency": 0.35,
        "budget_compliance": 0.25
    }
    
    overall_score = sum(scores[k] * weights[k] for k in scores)
    
    # Determine grade
    if overall_score >= 90:
        grade = "A+"
    elif overall_score >= 80:
        grade = "A"
    elif overall_score >= 70:
        grade = "B"
    elif overall_score >= 60:
        grade = "C"
    elif overall_score >= 50:
        grade = "D"
    else:
        grade = "F"
    
    return {
        "score": round(overall_score, 1),
        "grade": grade,
        "breakdown": scores,
        "message": get_score_message(overall_score),
        "improvement_tips": get_improvement_tips(scores)
    }

def get_score_message(score: float) -> str:
    if score >= 80:
        return "Excellent! You're a sustainability leader in AI usage."
    elif score >= 60:
        return "Good progress! Room for improvement in some areas."
    elif score >= 40:
        return "Fair. Consider optimizing your AI model choices."
    else:
        return "Needs improvement. Review model selection and usage patterns."

def get_improvement_tips(scores: dict) -> list:
    tips = []
    if scores.get("model_efficiency", 100) < 70:
        tips.append("Consider using smaller models for simple tasks (DistilBERT, LLaMA-8B)")
    if scores.get("energy_efficiency", 100) < 70:
        tips.append("Reduce token counts where possible, use caching for repeated queries")
    if scores.get("budget_compliance", 100) < 70:
        tips.append("Review usage patterns, set up alerts at lower thresholds")
    if not tips:
        tips.append("Keep up the great work! Consider mentoring other teams.")
    return tips


# ============================================================================
# 5. AI TRAINING EMISSIONS CALCULATOR
# Estimate emissions for training AI models
# ============================================================================

# Training energy estimates (kWh) based on model size and architecture
TRAINING_ENERGY_ESTIMATES = {
    # Model: kWh for full training run
    "gpt-4": 50000000,        # ~50 GWh (estimated)
    "gpt-3.5": 10000000,      # ~10 GWh
    "gpt-3": 1300000,         # ~1.3 GWh (published)
    "llama-3-70b": 2000000,   # ~2 GWh
    "llama-3-8b": 200000,     # ~200 MWh
    "llama-2-70b": 1500000,   # ~1.5 GWh
    "bert-base": 1500,        # ~1.5 MWh
    "bert-large": 4000,       # ~4 MWh
    "stable-diffusion-xl": 500000,  # ~500 MWh
    "custom-small": 10000,    # 10 MWh baseline
    "custom-medium": 100000,  # 100 MWh baseline
    "custom-large": 1000000,  # 1 GWh baseline
}

def estimate_training_emissions(
    model_type: str = "custom-medium",
    gpu_hours: int = None,
    gpu_type: str = "a100",
    num_gpus: int = 8,
    region: str = "global",
    parameters_billions: float = None
) -> dict:
    """Estimate CO2 emissions for model training."""
    
    # GPU power consumption (Watts)
    GPU_POWER = {
        "a100": 400,
        "h100": 700,
        "v100": 300,
        "a10": 150,
        "t4": 70,
        "4090": 450,
        "3090": 350
    }
    
    if gpu_hours:
        # Calculate from GPU hours
        gpu_power_w = GPU_POWER.get(gpu_type.lower(), 400)
        total_energy_wh = gpu_hours * num_gpus * gpu_power_w
        total_energy_kwh = total_energy_wh / 1000
    elif parameters_billions:
        # Estimate from model size (rough scaling law)
        # ~6 kWh per billion parameters per training epoch (very rough)
        training_epochs = 1
        base_energy = parameters_billions * 6000 * training_epochs
        total_energy_kwh = base_energy * 1.5  # Add overhead
    else:
        # Use preset
        total_energy_kwh = TRAINING_ENERGY_ESTIMATES.get(
            model_type.lower(), 
            TRAINING_ENERGY_ESTIMATES["custom-medium"]
        )
    
    # Apply PUE (data center overhead)
    total_energy_kwh *= 1.2
    
    # Carbon intensity
    from app.services.estimator import REGION_CI_TABLE
    ci = REGION_CI_TABLE.get(region.lower(), 475)
    
    # Calculate CO2
    co2_kg = (total_energy_kwh * ci) / 1000
    co2_tonnes = co2_kg / 1000
    
    # Context comparisons
    comparisons = {
        "equivalent_car_km": co2_kg / 0.21,
        "equivalent_flights_nyc_sf": co2_kg / 500,
        "equivalent_home_years": co2_kg / 8000,  # Average US home ~8 tonnes/year
        "equivalent_trees_lifetime": co2_kg / (21 * 40),  # 40 year lifetime
    }
    
    # Break-even analysis (how many inferences to match training)
    inference_co2_g = 0.1  # Approximate CO2 per inference
    break_even_queries = (co2_kg * 1000) / inference_co2_g
    
    return {
        "model_type": model_type,
        "total_energy_kwh": round(total_energy_kwh, 2),
        "total_energy_mwh": round(total_energy_kwh / 1000, 2),
        "co2_kg": round(co2_kg, 2),
        "co2_tonnes": round(co2_tonnes, 2),
        "region": region,
        "carbon_intensity": ci,
        "comparisons": comparisons,
        "break_even_queries": int(break_even_queries),
        "methodology": f"Estimated {total_energy_kwh:.0f} kWh × {ci} gCO2/kWh = {co2_tonnes:.1f} tonnes CO2",
        "offset_cost_usd": round(co2_tonnes * 25, 2)  # VCS price
    }


# ============================================================================
# 6. REPORT GENERATION
# Generate exportable reports
# ============================================================================

def generate_report(
    org_id: str = None,
    report_type: str = "monthly",
    format: str = "json"
) -> dict:
    """Generate a comprehensive carbon emissions report."""
    
    if org_id and org_id in ORGANIZATIONS:
        org = ORGANIZATIONS[org_id]
        org_logs = [l for l in USAGE_LOG if l["org_id"] == org_id]
    else:
        org = None
        org_logs = USAGE_LOG
    
    # Calculate period
    now = datetime.now()
    if report_type == "monthly":
        period_start = now.replace(day=1, hour=0, minute=0, second=0)
        period_name = now.strftime("%B %Y")
    elif report_type == "weekly":
        period_start = now - timedelta(days=7)
        period_name = f"Week of {period_start.strftime('%Y-%m-%d')}"
    else:
        period_start = now - timedelta(days=365)
        period_name = "Annual"
    
    # Filter logs by period
    period_logs = [l for l in org_logs if l["timestamp"] >= period_start.isoformat()]
    
    # Calculate totals
    total_co2_kg = sum(l["co2_kg"] for l in period_logs)
    total_energy_kwh = sum(l["energy_kwh"] for l in period_logs)
    total_queries = len(period_logs)
    
    # Breakdown by category
    by_type = {}
    by_model = {}
    by_day = {}
    
    for log in period_logs:
        mt = log["model_type"]
        by_type[mt] = by_type.get(mt, {"co2_kg": 0, "count": 0})
        by_type[mt]["co2_kg"] += log["co2_kg"]
        by_type[mt]["count"] += 1
        
        mn = log["model_name"]
        by_model[mn] = by_model.get(mn, 0) + log["co2_kg"]
        
        day = log["timestamp"][:10]
        by_day[day] = by_day.get(day, 0) + log["co2_kg"]
    
    # Get sustainability score
    score = calculate_sustainability_score(org_id)
    
    # Get offset requirements
    offset = calculate_offset_requirements(total_co2_kg)
    
    report = {
        "report_type": report_type,
        "period": period_name,
        "generated_at": now.isoformat(),
        "organization": org["name"] if org else "All Users",
        "summary": {
            "total_co2_kg": round(total_co2_kg, 4),
            "total_energy_kwh": round(total_energy_kwh, 6),
            "total_queries": total_queries,
            "avg_co2_per_query_g": round((total_co2_kg * 1000) / total_queries, 4) if total_queries > 0 else 0
        },
        "breakdown": {
            "by_model_type": by_type,
            "by_model": dict(sorted(by_model.items(), key=lambda x: x[1], reverse=True)[:10]),
            "daily_trend": by_day
        },
        "sustainability": {
            "score": score["score"],
            "grade": score["grade"],
            "improvement_tips": score["improvement_tips"]
        },
        "carbon_offset": {
            "trees_needed": offset["trees_needed_annual"],
            "offset_cost_usd": offset["cost_usd"],
            "equivalents": offset["real_world_equivalents"]
        },
        "recommendations": generate_org_recommendations(org_id) if org_id else []
    }
    
    return report

def generate_org_recommendations(org_id: str) -> list:
    """Generate personalized recommendations for an organization."""
    recommendations = []
    
    if org_id not in ORGANIZATIONS:
        return recommendations
    
    org_logs = [l for l in USAGE_LOG if l["org_id"] == org_id]
    
    if not org_logs:
        return ["Start tracking your AI usage to get personalized recommendations"]
    
    # Analyze patterns
    models_used = [l["model_name"] for l in org_logs]
    frontier_usage = sum(1 for m in models_used if m in ["gpt-4", "claude-3-opus", "sora"])
    
    if frontier_usage / len(models_used) > 0.5:
        recommendations.append({
            "type": "model_optimization",
            "priority": "high",
            "message": "Over 50% of queries use frontier models. Consider using smaller models for simpler tasks.",
            "potential_savings": "Up to 90% carbon reduction"
        })
    
    # Check for video usage
    video_usage = sum(1 for l in org_logs if l["model_type"] == "video")
    if video_usage > 0:
        recommendations.append({
            "type": "video_awareness",
            "priority": "medium",
            "message": "Video generation is very carbon-intensive. Consider shorter clips or lower resolutions.",
            "potential_savings": "50-80% reduction with lower resolution"
        })
    
    # Regional optimization
    recommendations.append({
        "type": "regional_optimization",
        "priority": "medium",
        "message": "Consider routing workloads to low-carbon regions (EU-North, US-West) during peak times.",
        "potential_savings": "Up to 60% carbon reduction"
    })
    
    # Time-shifting
    recommendations.append({
        "type": "time_shifting",
        "priority": "low",
        "message": "Schedule batch workloads during solar peak hours (11am-3pm) for lower carbon intensity.",
        "potential_savings": "10-30% carbon reduction"
    })
    
    return recommendations


# ============================================================================
# 7. COMPARATIVE ANALYTICS
# Compare your usage against benchmarks
# ============================================================================

# Industry benchmarks (kg CO2 per 1000 AI queries)
INDUSTRY_BENCHMARKS = {
    "tech_startup": 0.5,
    "enterprise": 2.0,
    "research_lab": 5.0,
    "media_production": 10.0,
    "ai_company": 15.0
}

def get_comparative_analytics(org_id: str, industry: str = "tech_startup") -> dict:
    """Compare organization's usage against industry benchmarks."""
    
    if org_id not in ORGANIZATIONS:
        return {"error": "Organization not found"}
    
    org_logs = [l for l in USAGE_LOG if l["org_id"] == org_id]
    
    if not org_logs:
        return {"message": "Not enough data for comparison"}
    
    total_co2_kg = sum(l["co2_kg"] for l in org_logs)
    total_queries = len(org_logs)
    
    # Calculate per-1000-query metric
    co2_per_1k = (total_co2_kg / total_queries) * 1000 if total_queries > 0 else 0
    
    benchmark = INDUSTRY_BENCHMARKS.get(industry, 2.0)
    
    comparison_percent = (co2_per_1k / benchmark) * 100
    
    if comparison_percent < 50:
        status = "excellent"
        message = "You're using less than half the industry average!"
    elif comparison_percent < 100:
        status = "good"
        message = "You're below the industry benchmark."
    elif comparison_percent < 150:
        status = "average"
        message = "You're close to the industry average."
    else:
        status = "needs_improvement"
        message = "You're above the industry benchmark. Consider optimizations."
    
    return {
        "organization_id": org_id,
        "industry": industry,
        "metrics": {
            "total_queries": total_queries,
            "total_co2_kg": round(total_co2_kg, 4),
            "co2_per_1k_queries": round(co2_per_1k, 4),
            "industry_benchmark": benchmark
        },
        "comparison": {
            "percent_of_benchmark": round(comparison_percent, 1),
            "status": status,
            "message": message
        },
        "ranking": {
            "all_industries": INDUSTRY_BENCHMARKS,
            "your_position": get_industry_rank(co2_per_1k)
        }
    }

def get_industry_rank(co2_per_1k: float) -> str:
    """Determine which industry tier the usage falls into."""
    sorted_industries = sorted(INDUSTRY_BENCHMARKS.items(), key=lambda x: x[1])
    for industry, benchmark in sorted_industries:
        if co2_per_1k <= benchmark:
            return f"Better than {industry} average"
    return "Above all benchmarks"


# ============================================================================
# 8. ALERT SYSTEM
# Configurable alerts for budget and usage
# ============================================================================

ALERT_CONFIGS = {}

def configure_alerts(
    org_id: str,
    budget_threshold_percent: int = 80,
    daily_limit_kg: float = None,
    model_restrictions: list = None,
    email_notifications: bool = True
) -> dict:
    """Configure alert settings for an organization."""
    
    ALERT_CONFIGS[org_id] = {
        "budget_threshold_percent": budget_threshold_percent,
        "daily_limit_kg": daily_limit_kg,
        "model_restrictions": model_restrictions or [],
        "email_notifications": email_notifications,
        "configured_at": datetime.now().isoformat()
    }
    
    if org_id in ORGANIZATIONS:
        ORGANIZATIONS[org_id]["alert_threshold_percent"] = budget_threshold_percent
    
    return {
        "status": "configured",
        "settings": ALERT_CONFIGS[org_id]
    }

def check_alerts(org_id: str, model_name: str = None, co2_kg: float = 0) -> list:
    """Check if any alerts should be triggered."""
    alerts = []
    
    if org_id not in ORGANIZATIONS:
        return alerts
    
    org = ORGANIZATIONS[org_id]
    config = ALERT_CONFIGS.get(org_id, {})
    
    # Budget threshold alert
    usage_percent = (org["current_month_usage_kg"] / org["monthly_budget_kg"]) * 100
    threshold = config.get("budget_threshold_percent", 80)
    
    if usage_percent >= threshold:
        alerts.append({
            "type": "budget_threshold",
            "severity": "warning" if usage_percent < 100 else "critical",
            "message": f"Monthly budget at {usage_percent:.1f}%",
            "action": "Consider reducing AI usage or upgrading budget"
        })
    
    # Daily limit alert
    daily_limit = config.get("daily_limit_kg")
    if daily_limit:
        today = datetime.now().strftime("%Y-%m-%d")
        today_usage = sum(
            l["co2_kg"] for l in USAGE_LOG 
            if l["org_id"] == org_id and l["timestamp"].startswith(today)
        )
        if today_usage >= daily_limit:
            alerts.append({
                "type": "daily_limit",
                "severity": "warning",
                "message": f"Daily limit of {daily_limit}kg reached ({today_usage:.2f}kg used)",
                "action": "Defer non-critical workloads to tomorrow"
            })
    
    # Model restriction alert
    restricted = config.get("model_restrictions", [])
    if model_name and model_name.lower() in [m.lower() for m in restricted]:
        alerts.append({
            "type": "model_restricted",
            "severity": "info",
            "message": f"Model '{model_name}' is on the restricted list",
            "action": "Consider using an approved alternative model"
        })
    
    return alerts


# ============================================================================
# 9. API USAGE ANALYTICS
# Track and analyze API endpoint usage
# ============================================================================

API_USAGE_LOG = []

def log_api_call(
    endpoint: str,
    method: str,
    user_id: str = None,
    org_id: str = None,
    response_time_ms: float = 0,
    co2_grams: float = 0
):
    """Log an API call for analytics."""
    API_USAGE_LOG.append({
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "method": method,
        "user_id": user_id,
        "org_id": org_id,
        "response_time_ms": response_time_ms,
        "co2_grams": co2_grams
    })

def get_api_analytics(hours: int = 24) -> dict:
    """Get API usage analytics for the specified time period."""
    cutoff = datetime.now() - timedelta(hours=hours)
    recent_calls = [
        l for l in API_USAGE_LOG 
        if datetime.fromisoformat(l["timestamp"]) >= cutoff
    ]
    
    total_calls = len(recent_calls)
    total_co2 = sum(l["co2_grams"] for l in recent_calls)
    avg_response_time = (
        sum(l["response_time_ms"] for l in recent_calls) / total_calls 
        if total_calls > 0 else 0
    )
    
    # Group by endpoint
    by_endpoint = {}
    for call in recent_calls:
        ep = call["endpoint"]
        by_endpoint[ep] = by_endpoint.get(ep, {"calls": 0, "co2": 0})
        by_endpoint[ep]["calls"] += 1
        by_endpoint[ep]["co2"] += call["co2_grams"]
    
    # Hourly distribution
    hourly = {}
    for call in recent_calls:
        hour = call["timestamp"][11:13]
        hourly[hour] = hourly.get(hour, 0) + 1
    
    return {
        "period_hours": hours,
        "total_calls": total_calls,
        "total_co2_grams": round(total_co2, 4),
        "avg_response_time_ms": round(avg_response_time, 2),
        "calls_per_hour": round(total_calls / hours, 1) if hours > 0 else 0,
        "by_endpoint": by_endpoint,
        "hourly_distribution": hourly
    }


# ============================================================================
# 10. EMISSIONS FORECASTING
# Predict future emissions based on trends
# ============================================================================

def forecast_emissions(
    org_id: str = None,
    days_ahead: int = 30
) -> dict:
    """Forecast future emissions based on historical trends."""
    
    if org_id and org_id in ORGANIZATIONS:
        org_logs = [l for l in USAGE_LOG if l["org_id"] == org_id]
    else:
        org_logs = USAGE_LOG
    
    if len(org_logs) < 7:
        return {
            "message": "Need at least 7 days of data for forecasting",
            "data_points": len(org_logs)
        }
    
    # Calculate daily averages from historical data
    daily_totals = {}
    for log in org_logs:
        day = log["timestamp"][:10]
        daily_totals[day] = daily_totals.get(day, 0) + log["co2_kg"]
    
    if not daily_totals:
        return {"message": "No data available for forecasting"}
    
    avg_daily_co2 = sum(daily_totals.values()) / len(daily_totals)
    
    # Simple trend detection (compare recent week to overall)
    sorted_days = sorted(daily_totals.keys())
    recent_days = sorted_days[-7:] if len(sorted_days) >= 7 else sorted_days
    recent_avg = sum(daily_totals[d] for d in recent_days) / len(recent_days)
    
    trend = "stable"
    trend_percent = ((recent_avg - avg_daily_co2) / avg_daily_co2) * 100 if avg_daily_co2 > 0 else 0
    
    if trend_percent > 10:
        trend = "increasing"
    elif trend_percent < -10:
        trend = "decreasing"
    
    # Forecast
    forecast_daily = recent_avg * (1 + trend_percent / 100 * 0.1)  # Dampen trend
    forecast_total = forecast_daily * days_ahead
    
    # Generate daily forecasts
    daily_forecast = []
    for i in range(days_ahead):
        day = (datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        # Add some variation
        day_forecast = forecast_daily * (1 + random.uniform(-0.1, 0.1))
        daily_forecast.append({
            "date": day,
            "forecast_co2_kg": round(day_forecast, 4)
        })
    
    return {
        "forecast_period_days": days_ahead,
        "historical_avg_daily_kg": round(avg_daily_co2, 4),
        "recent_avg_daily_kg": round(recent_avg, 4),
        "trend": trend,
        "trend_percent": round(trend_percent, 1),
        "forecast_daily_avg_kg": round(forecast_daily, 4),
        "forecast_total_kg": round(forecast_total, 4),
        "daily_forecast": daily_forecast[:7],  # First 7 days
        "confidence": "medium" if len(org_logs) >= 30 else "low"
    }
