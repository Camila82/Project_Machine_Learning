# Urban Noise Analysis System — Colombian Cities
## CRISP-ML Machine Learning Project

A Flask web application following the CRISP-ML methodology for analyzing
urban noise pollution across 8 major Colombian cities using open sensor data
and mobility indicators.

---

## Project Structure

urban-noise-colombia/
├── app.py
├── requirements.txt
├── Procfile
├── pipeline/
├── data/raw/
├── data/processed/
├── routes/
├── utils/
├── templates/
└── static/

## Setup Instructions

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd urban-noise-colombia

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the data pipeline

```bash
python pipeline/data_engineering.py
```

Expected:
7 JSON files created inside `data/processed/`

### 3. Run Flask

```bash
python app.py
```

Open:
http://localhost:5000

---

## Dataset

- Source: Urban noise sensor measurements
- Records: 1,000
- Cities: Bogotá, Medellín, Cali, Barranquilla, Cartagena, Bucaramanga, Manizales, Pereira

## Methodology

CRISP-ML

## Academic Context

University Machine Learning Project — 2025