from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os
import sys
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import BurnoutPredictor

app = FastAPI(title="Burnout Risk Prediction API")

# Path Setup
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")
web_dir = os.path.join(base_dir, "src", "web")

# Initialize Predictor
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        if not os.path.exists(os.path.join(models_dir, "models/svm_model.joblib")):
            raise RuntimeError("Model files not found. Please run train.py first.")
        predictor = BurnoutPredictor(models_dir)
    return predictor

class EmployeeMetrics(BaseModel):
    Age: int
    Gender: str
    Country: str
    Job_Role: str
    Experience_Years: int
    Company_Size: str
    Work_Hours_Per_Day: float
    Meetings_Per_Day: int
    Internet_Speed_Mbps: float
    Work_Environment: str
    Sleep_Hours: float
    Exercise_Hours_Per_Week: float
    Screen_Time_Hours: float
    Stress_Level: str
    Productivity_Score: int

# Serve static files
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/")
async def root():
    # Serve index.html as the root
    index_path = os.path.join(web_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Burnout Risk Prediction API is running", "version": "1.0.0"}

@app.get("/style.css")
async def get_css():
    return FileResponse(os.path.join(web_dir, "style.css"))

@app.get("/app.js")
async def get_js():
    return FileResponse(os.path.join(web_dir, "app.js"))

@app.post("/predict")
async def predict(metrics: EmployeeMetrics):
    try:
        p = get_predictor()
        # Convert Pydantic model to dict
        data = metrics.dict()
        prediction = p.predict(data)[0]
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)