from fastapi import FastAPI
import tensorflow as tf
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Paksa pakai CPU

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    print("⏳ Loading model...")
    start = time.time()
    model = tf.keras.models.load_model("best_model.keras")
    print(f"✅ Model loaded in {time.time() - start:.2f} seconds.")

@app.get("/")
def read_root():
    return {
        "message": "Hello, FastAPI!",
        "model_loaded": model is not None
    }
