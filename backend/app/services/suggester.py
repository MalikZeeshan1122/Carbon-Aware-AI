
# Right-sizing recommendation engine (2026 Models)
# Maps task complexity to appropriate model tiers

TASK_MODEL_MAPPING = {
    # Simple tasks -> Micro/Efficient models
    "classification": {
        "recommended": "distilbert",
        "alternatives": ["phi-3-mini", "gemma-2-9b"],
        "tier": "micro",
        "reason": "Classification tasks can be handled by specialized lightweight models like DistilBERT with >95% lower energy."
    },
    "sentiment analysis": {
        "recommended": "distilbert",
        "alternatives": ["phi-3-mini"],
        "tier": "micro",
        "reason": "Sentiment analysis is a solved problem for small models. DistilBERT achieves 97% of BERT's accuracy at 60% less compute."
    },
    "spam detection": {
        "recommended": "distilbert",
        "alternatives": ["phi-3-mini"],
        "tier": "micro",
        "reason": "Spam detection is a binary classification task perfect for lightweight classifiers."
    },
    
    # Moderate tasks -> Efficient/Standard models
    "summarization": {
        "recommended": "gemma-2-9b",
        "alternatives": ["claude-3.5-haiku", "gpt-4o-mini"],
        "tier": "efficient",
        "reason": "Gemma 2 9B excels at summarization while using 90% less energy than frontier models."
    },
    "translation": {
        "recommended": "gemma-2-9b",
        "alternatives": ["gpt-4o-mini", "claude-3.5-haiku"],
        "tier": "efficient",
        "reason": "Translation is well-handled by efficient models like Gemma 2."
    },
    "extraction": {
        "recommended": "llama-4-8b",
        "alternatives": ["phi-4", "mistral-small"],
        "tier": "efficient",
        "reason": "Information extraction works well with 8B parameter models, no need for frontier models."
    },
    "simple q&a": {
        "recommended": "llama-4-8b",
        "alternatives": ["phi-4", "gpt-4o-mini"],
        "tier": "efficient",
        "reason": "Simple Q&A doesn't require the reasoning capabilities of large models."
    },
    
    # Creative tasks -> Standard models
    "creative writing": {
        "recommended": "llama-4-70b",
        "alternatives": ["claude-3.5-sonnet", "gemini-2.0-flash"],
        "tier": "standard",
        "reason": "Creative writing benefits from larger models, but open-source LLaMA 4 70B matches GPT-4 at lower cost."
    },
    "code generation": {
        "recommended": "deepseek-coder-v2",
        "alternatives": ["codestral", "qwen-2.5-coder-32b"],
        "tier": "standard",
        "reason": "Specialized code models like DeepSeek Coder v2 outperform general models while using less energy."
    },
    
    # Complex tasks -> May need Frontier (but with warning)
    "reasoning": {
        "recommended": "o1-mini",
        "alternatives": ["deepseek-r1", "o1"],
        "tier": "frontier",
        "reason": "Complex reasoning benefits from specialized reasoning models. o1-mini provides good balance of capability and efficiency."
    },
    "complex analysis": {
        "recommended": "claude-3.5-sonnet",
        "alternatives": ["gpt-4o", "gemini-2.0-pro"],
        "tier": "frontier",
        "reason": "Deep analysis tasks may warrant frontier models. Consider breaking into smaller subtasks."
    }
}

# Energy tiers for comparison
TIER_ENERGY_MULTIPLIER = {
    "micro": 1,
    "efficient": 3,
    "standard": 10,
    "frontier": 50
}

def recommend_model(task_type: str, current_model: str) -> dict:
    """
    Recommend a more efficient model based on task complexity.
    Returns recommendation with savings estimate.
    """
    task = task_type.lower().strip()
    current = current_model.lower().strip()
    
    # Default response
    recommendation = current
    reason = "Your current model is appropriate for this task."
    savings = 0.0
    
    # Try to find task in mapping
    task_info = TASK_MODEL_MAPPING.get(task)
    
    if task_info:
        recommended = task_info["recommended"]
        
        # Check if current model is less efficient than recommended
        current_tier = _get_model_tier(current)
        recommended_tier = task_info["tier"]
        
        current_multiplier = TIER_ENERGY_MULTIPLIER.get(current_tier, 10)
        recommended_multiplier = TIER_ENERGY_MULTIPLIER.get(recommended_tier, 10)
        
        if current_multiplier > recommended_multiplier:
            recommendation = recommended
            reason = task_info["reason"]
            savings = round((1 - recommended_multiplier / current_multiplier) * 100, 1)
    else:
        # Unknown task - give general advice if using frontier model
        if current in ["gpt-4", "claude-3-opus", "gemini-ultra"]:
            recommendation = "llama-3-70b"
            reason = "Consider if your task truly needs a frontier model. LLaMA 3 70B is a strong alternative for most use cases."
            savings = 80.0
    
    return {
        "recommended_model": recommendation,
        "savings_percentage": savings,
        "reason": reason
    }

def _get_model_tier(model_name: str) -> str:
    """Determine the tier of a model based on its name."""
    model = model_name.lower()
    
    if model in ["gpt-4", "gpt-4-turbo", "claude-3-opus", "gemini-ultra"]:
        return "frontier"
    elif model in ["gpt-3.5-turbo", "claude-3-sonnet", "mistral-large", "llama-3-70b", "llama-2-70b"]:
        return "standard"
    elif model in ["llama-3-8b", "llama-2-13b", "llama-2-7b", "mistral-medium", "flan-t5-large", "claude-3-haiku"]:
        return "efficient"
    else:
        return "micro"

def get_task_types():
    """Return list of all supported task types."""
    return list(TASK_MODEL_MAPPING.keys())
