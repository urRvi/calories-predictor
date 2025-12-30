from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Calories Burnt Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), '../src/models/xgboost_model.json')
scaler_path = os.path.join(os.path.dirname(__file__), '../src/models/scaler.pkl')
label_encoders_path = os.path.join(os.path.dirname(__file__), '../src/models/label_encoders.pkl')

model = xgb.XGBRegressor()
model.load_model(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(label_encoders_path)

class PredictionRequest(BaseModel):
    Gender: str
    Age: int
    Height: float
    Weight: float
    Duration: float
    Heart_Rate: float
    Body_Temp: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Calories Burnt Prediction API"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Prepare input data
        data = request.dict()
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        if 'Gender' in df.columns:
            if 'Gender' in label_encoders:
                le = label_encoders['Gender']
                # Handle unseen labels if necessary, but for Gender it's likely just Male/Female
                try:
                    df['Gender'] = le.transform(df['Gender'])
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid Gender value. Expected {list(le.classes_)}")
            else:
                 raise HTTPException(status_code=500, detail="Label encoder for Gender not found")

        # Scale features        
        
        features = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        df = df[features]
        
        scaled_data = scaler.transform(df)
        
        # Predict
        prediction = model.predict(scaled_data)
        
        return {"calories_burnt": float(prediction[0])}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
