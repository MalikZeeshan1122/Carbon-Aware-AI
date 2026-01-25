# Carbon-Aware AI

A decision-support system to estimate and reduce the carbon footprint of AI queries.

## Project Structure

- `backend/`: FastAPI application for logic and estimation.
- `frontend/`: Streamlit dashboard for user interaction.

## How to Run

### Prerequisities
- Python 3.9+
- pip

### 1. Setup Backend

```bash
cd backend
pip install -r requirements.txt
python -m app.main
```

The API will run at `http://localhost:8000`. Documentation at `http://localhost:8000/docs`.

### 2. Setup Frontend

Open a new terminal:
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

The Dashboard will open at `http://localhost:8501`.

## Features
- **Estimate Emissions**: Enter model and token counts to see CO2 impact.
- **Right-Sizing**: Get recommendations for smaller models based on task type.
