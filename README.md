# Calories Burnt Prediction

An advanced Machine Learning application to predict calories burnt during exercise. This project features a high-accuracy XGBoost model served via a FastAPI backend and a premium dark-mode frontend.

## Features
- **Machine Learning**: XGBoost model with ~99.9% R2 score.
- **Deep Learning**: PyTorch model implementation included for comparison.
- **Backend**: Fast and efficient API using FastAPI.
- **Frontend**: Modern, responsive web interface with dark mode and animations.
- **Data**: Trained on 15,000+ samples of exercise data.

## Quick Start
**Windows Users:**
Simply double-click `run_app.bat` to start the application!

**Manual Start:**
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the API server:
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```
3. In a separate terminal, start the frontend:
   ```bash
   cd frontend && python -m http.server 3000
   ```
4. Open `http://localhost:3000` in your browser

## Project Structure
- `api/`: FastAPI backend code.
- `frontend/`: HTML/CSS/JS frontend code.
- `src/`: Model training and preprocessing logic.
- `notebooks/`: EDA and experimentation notebooks.
- `data/`: Dataset files.
- `scripts/`: Utility scripts.

## Tech Stack
- **Python**: 3.8+
- **ML**: XGBoost, PyTorch, Scikit-learn
- **Web**: FastAPI, HTML5, CSS3, JavaScript
- **Data**: Pandas, NumPy, Matplotlib, Seaborn
