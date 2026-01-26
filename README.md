# 🌱 Carbon-Aware AI

**Make the Climate Impact of AI Visible, Measurable, and Actionable**

A comprehensive decision-support system to estimate, track, and reduce the carbon footprint of AI operations including text generation, image creation, and video synthesis.

---

## 🎯 Overview

Carbon-Aware AI helps developers, researchers, and organizations understand and minimize the environmental impact of their AI workloads. The platform provides:

- **Real-time carbon footprint estimation** for AI models
- **Smart model recommendations** to reduce emissions
- **Multi-modal support**: Text AI, Image Generation, and Video Generation
- **Data center comparison** across global cloud providers
- **Historical tracking** of your AI carbon footprint

---

## ✨ Features

### 📊 Text AI Carbon Estimation
- Estimate CO₂ emissions for LLM queries (GPT-4, Claude, Llama, etc.)
- Support for 9+ popular AI models
- Token-based calculation with regional carbon intensity
- Batch query analysis

### 🖼️ Image Generation Tracking
- Track emissions for DALL-E, Stable Diffusion, Midjourney, Flux, and more
- Resolution and step-based calculations
- Model comparison visualizations

### 🎬 Video Generation Analysis
- Measure carbon footprint for Sora, Runway, Pika, and other video models
- Duration, resolution, and FPS-based estimates
- Real-world equivalents (phone charges, streaming hours, car emissions)

### 🏢 Data Center Insights
- Compare carbon intensity across global data centers
- PUE (Power Usage Effectiveness) metrics
- Renewable energy percentage tracking
- Provider-level emissions analysis

### 📈 History & Analytics
- Track your carbon footprint over time
- Visualize trends and patterns
- Export data for reporting

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MalikZeeshan1122/Carbon-Aware-AI.git
cd Carbon-Aware-AI
```

2. **Set up the Backend API**

Open a terminal and run:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API will be available at:
- **API Endpoint**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

3. **Set up the Frontend Dashboard**

Open a **new terminal** and run:
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

The dashboard will open automatically at **http://localhost:8501**

---

## 📁 Project Structure

```
Carbon-Aware-AI/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # API entry point
│   │   ├── models/         # Data models
│   │   ├── services/       # Business logic
│   │   │   ├── estimator.py      # Carbon estimation engine
│   │   │   ├── recommender.py    # Model recommendation system
│   │   │   ├── image_estimator.py
│   │   │   ├── video_estimator.py
│   │   │   └── datacenter.py
│   │   └── utils/          # Helper functions
│   └── requirements.txt
│
├── frontend/               # Streamlit dashboard
│   ├── app.py             # Main dashboard application
│   └── requirements.txt
│
├── README.md
└── LICENSE
```

---

## 🎮 Usage

### Estimating Text AI Emissions

1. Open the dashboard at http://localhost:8501
2. Select your AI model (e.g., GPT-4, Claude, Llama)
3. Choose the task type (Classification, Summarization, etc.)
4. Set prompt and completion token counts
5. Select your cloud region
6. Click **"🌿 Estimate Impact"**

You'll see:
- Energy consumed (kWh)
- CO₂ emissions (grams)
- Real-world equivalents (phone charges, tree-days needed)
- Smart recommendations for more efficient models

### Image Generation Analysis

1. Navigate to the **"🖼️ Image AI"** tab
2. Select your image model (DALL-E, Stable Diffusion, etc.)
3. Configure resolution, steps, and number of images
4. Click **"🌿 Estimate Image Impact"**

### Video Generation Tracking

1. Go to the **"🎬 Video AI"** tab
2. Choose your video model (Sora, Runway, etc.)
3. Set duration, resolution, and FPS
4. Click **"🌿 Estimate Video Impact"**

---

## 🔧 API Endpoints

### Text AI
- `POST /estimate` - Estimate carbon footprint for text generation
- `POST /recommend` - Get model recommendations

### Image AI
- `GET /image/estimate` - Estimate image generation emissions

### Video AI
- `GET /video/estimate` - Estimate video generation emissions

### Data Centers
- `GET /datacenters/comparison` - Compare global data centers

For full API documentation, visit http://localhost:8000/docs

---

## 📊 Supported Models

### Text Models
- GPT-4, GPT-3.5-turbo (OpenAI)
- Claude-3-Opus, Claude-3-Sonnet (Anthropic)
- Llama-3-70B, Llama-3-8B (Meta)
- Mistral-Large (Mistral AI)
- DistilBERT, FLAN-T5 (Hugging Face/Google)

### Image Models
- DALL-E 2 & 3, Stable Diffusion (XL, 3, 2.1, Turbo, Lightning)
- Midjourney v5 & v6, Flux (Pro, Schnell)
- Imagen 2, LCM

### Video Models
- Sora, Runway Gen2 & Gen3, Veo
- Pika 1.0 & 1.5, Stable Video Diffusion, AnimateDiff

---

## 🌍 Regional Carbon Intensity

The system accounts for different carbon intensities across regions:
- **Global** (average)
- **US-East** (Virginia)
- **US-West** (Oregon)
- **EU-West** (Ireland)
- **AsiaPac** (Singapore)

---

## 🧮 Methodology

Our carbon estimation is based on:
1. **Model parameters** and computational requirements
2. **Token/pixel/frame counts** for workload estimation
3. **Regional carbon intensity** (gCO₂/kWh)
4. **Data center PUE** (Power Usage Effectiveness)
5. **Hardware efficiency** (GPU/TPU specifications)

References:
- Patterson et al. (2021) - "Carbon Emissions and Large Neural Network Training"
- Strubell et al. (2019) - "Energy and Policy Considerations for Deep Learning in NLP"
- Luccioni et al. (2023) - "Estimating the Carbon Footprint of BLOOM"

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Carbon intensity data from [Electricity Maps](https://www.electricitymaps.com/)
- Model efficiency research from leading AI labs
- Open-source community for tools and libraries

---

## 📧 Contact

**Malik Zeeshan**
- GitHub: [@MalikZeeshan1122](https://github.com/MalikZeeshan1122)
- Repository: [Carbon-Aware-AI](https://github.com/MalikZeeshan1122/Carbon-Aware-AI)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

**Together, let's make AI more sustainable! 🌱**
