from fastapi import FastAPI, UploadFile, File
import uvicorn
from prediction_api import createBoundingBox
import numpy as np
import cv2

app = FastAPI()

@app.get('/')
async def COT():
    return "Welcome to Coral-of-Thorns Starfish Object Detection"

@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):

    # calls read_image function from prediction_api.py
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
    print("Image loaded successfully")

    # make predictions
    results = createBoundingBox(image, threshold=0.5)

    #for i in range(len(results)):
    #    return{
    #        "class": results[i]["class"],
    #        "class_name": results[i]["class_name"],
    #        "bbox": results[i]["bbox"], 
    #        "confidence": results[i]["confidence"]
    #} 

    return results

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host='0.0.0.0')