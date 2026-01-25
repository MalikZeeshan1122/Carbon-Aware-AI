# Carbon-Aware AI: Making the Climate Impact of AI Visible

**Author:** [Your Name]  
**Project Type:** Software Prototype / Decision Support System

## 1. Problem
Artificial Intelligence, particularly Large Language Models (LLMs), has a hidden but significant environmental cost. A single query to a model like GPT-4 can consume 10-100x more energy than a standard search query. Developers and organizations often default to the largest, most powerful models (e.g., "frontier models") for all tasks, even simple ones, leading to massive unnecessary CO₂ emissions. There is currently a lack of visibility and actionable data at the moment of decision-making.

## 2. Solution
**Carbon-Aware AI** is a decision-support system designed to reduce the carbon footprint of AI workflows by recommending "right-sized" models.

### Key Features
1.  **Real-time Emissions Estimation:** A FastAPI backend estimates energy (kWh) and CO₂ (grams) per query based on token count, model architecture, and regional carbon intensity.
2.  **Smart Right-Sizing Recommendations:** An intelligent logic layer analyzes the user's task type (e.g., summarization vs. complex reasoning) and suggests the smallest, most efficient model that maintains quality.
3.  **Interactive Dashboard:** A Streamlit-based frontend acts as a "Nutrition Label" for AI tasks, visualizing the trade-offs between model choice and environmental impact.

## 3. Methodology & Assumptions
-   **Coefficient-Based Estimation:** Uses energy-per-token coefficients derived from academic literature (Luccioni et al., 2023) to proxy inference energy.
-   **Carbon Intensity:** Integrates regional power grid data to show how *where* you run AI matters as much as *what* you run.
-   **Proportionality Principle:** The core recommendation engine is built on the principle that simple tasks should not use "Power-Hungry" models.

## 4. Impact & Ethics
-   **Behavioral Change:** By making the invisible visible, this tool aims to shift user behavior toward sustainable AI practices.
-   **Transparency:** Open-source assumptions allow for scrutiny and improvement of the estimation logic.
-   **Equity:** Encourages the use of smaller, open-source models (like Llama-3-8b) which are more accessible and efficient.

## 5. Project Status
-   [x] **Backend API:** Functioning FastAPI service for estimation and recommendation.
-   [x] **Frontend:** Interactive Streamlit dashboard for demonstrating the concept.
-   [x] **Logic:** Implemented initial lookup tables for energy coefficients and right-sizing rules.

**Next Steps:** Validate with real-world trace data and integrate dynamic carbon intensity APIs (e.g., ElectricityMaps).
