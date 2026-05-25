import os, requests, tensorflow as tf, io, uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

MODEL_URL = "YOUR_DIRECT_DOWNLOAD_LINK"  # <-- REPLACE
MODEL_PATH = "cat_dog_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(img):
    img = img.resize((128,128))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Cat/Dog API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inp = preprocess(img)
    pred = model.predict(inp)[0][0]
    return JSONResponse({"prediction": "Dog" if pred > 0.5 else "Cat"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
