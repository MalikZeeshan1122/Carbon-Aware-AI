import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.services.estimator import estimate_emissions
from app.services.suggester import recommend_model

def test_logic():
    print("Testing Estimator...")
    est = estimate_emissions("gpt-4", 500, 200)
    print(f"GPT-4 Estimate: {est}")
    assert est['co2_grams'] > 0
    
    print("\nTesting Suggester...")
    rec = recommend_model("classification", "gpt-4")
    print(f"Classification Recommendation: {rec}")
    assert rec['recommended_model'] == "distilbert"
    
    print("\nAll logic tests passed!")

if __name__ == "__main__":
    test_logic()
