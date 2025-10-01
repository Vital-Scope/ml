from fastapi import FastAPI
import numpy as np

from predict import analyze_folder

app = FastAPI(title="ML Service", version="1.0")

@app.get("/api/predict/")
async def get_predict(guid: str):
    result = analyze_folder(f"/proxy/{guid}")
    return str(np.float32(result))
