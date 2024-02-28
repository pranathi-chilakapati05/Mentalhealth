import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException

sm_pipeline = joblib.load('sm_lr_pipeline.joblib')
st_pipeline = joblib.load('lr_pipeline.joblib')

def predict_mental_health(params_dict,pipeline):
    data = pd.DataFrame(params_dict, index=[0])
    predictions = pipeline.predict(data)
    prob = pipeline.predict_proba(data)
    return predictions[0],np.max(prob)

app = FastAPI()

@app.post("/student/")
async def process_input(param_dict: dict):
    pred,prob = predict_mental_health(param_dict,st_pipeline)
    d = {}
    d['pred'] = pred
    d['prob'] = prob
    return d

@app.post("/social_media/")
async def process_input(param_dict: dict):
    pred,prob = predict_mental_health(param_dict,sm_pipeline)
    d = {}
    d['pred'] = pred
    d['prob'] = prob
    return d






