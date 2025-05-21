from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
import numpy as np
import librosa
import os
import time
from skimage.transform import resize
import tempfile
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

app = FastAPI()
model = None

note_classes = ['A2', 'A3', 'A4', 'Asharp2', 'Asharp3', 'Asharp4', 'B2', 'B3', 'B4', 'C3', 'C4', 'C5',
          'Csharp3', 'Csharp4', 'Csharp5', 'D2', 'D3', 'D4', 'D5', 'Dsharp2', 'Dsharp3', 'Dsharp4',
          'Dsharp5', 'E2', 'E3', 'E4', 'E5', 'F2', 'F3', 'F4', 'F5', 'Fsharp2', 'Fsharp3', 'Fsharp4',
          'Fsharp5', 'G2', 'G3', 'G4', 'G5', 'Gsharp2', 'Gsharp3', 'Gsharp4', 'Gsharp5']

chord_classes = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G',]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def load_and_segment_audio(file_path, segment_duration=0.50, sr=44100, auto_segment=False):
    audio_data, _ = librosa.load(file_path, sr=sr)
    
    if auto_segment:
        # Detect onsets for automatic segmentation
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr, 
                                                 wait=0.2, pre_avg=0.2, post_avg=0.2, 
                                                 pre_max=0.2, post_max=0.2)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Add end time
        onset_times = np.append(onset_times, len(audio_data)/sr)
        
        # Create segments based on detected onsets
        audio_segments = []
        for i in range(len(onset_times)-1):
            start_sample = int(onset_times[i] * sr)
            end_sample = int(onset_times[i+1] * sr)
            segment = audio_data[start_sample:end_sample]
            
            # Ensure consistent length by zero-padding short segments
            target_length = int(segment_duration * sr)
            if len(segment) < target_length:
                segment = np.pad(segment, (0, target_length - len(segment)), 'constant')
            
            # Split too long segments
            if len(segment) > target_length:
                split_segments = librosa.util.frame(segment, frame_length=target_length, hop_length=target_length).T
                for split_segment in split_segments:
                    audio_segments.append(split_segment)
            else:
                audio_segments.append(segment)
                
        audio_segments = np.array(audio_segments)
    else:
        # Original fixed-length segmentation
        audio_segments = librosa.util.frame(audio_data, frame_length=int(segment_duration*sr), hop_length=int(segment_duration*sr)).T
    
    if auto_segment:
        segment_times = onset_times[:-1]
        # Handle split segments
        adjusted_segment_times = []
        current_idx = 0
        for i in range(len(onset_times)-1):
            duration = onset_times[i+1] - onset_times[i]
            segments_count = int(np.ceil(duration / segment_duration))
            for j in range(segments_count):
                if current_idx < len(audio_segments):
                    adjusted_segment_times.append(onset_times[i] + j * segment_duration)
                    current_idx += 1
        segment_times = np.array(adjusted_segment_times)
    else:
        segment_times = np.arange(len(audio_segments)) * segment_duration
    
    return audio_segments, segment_times

def segments_to_mel_spectrograms(segments, sr=44100, target_shape=(128, 128)):
    mel_spectrograms = []
    for segment in segments:
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mel_spectrograms.append(mel_spectrogram)
    return np.array(mel_spectrograms)


@app.on_event("startup")
def load_model():
    global model
    print("⏳ Loading model...")
    start = time.time()
    try:
        model = tf.keras.models.load_model("best_model.keras")
        print(f"✅ Model loaded in {time.time() - start:.2f} seconds.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.get("/uploud", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r") as file:
        return file.read()
    
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/main.html", "r") as file:
        return file.read()

@app.get("/live", response_class=HTMLResponse)
def live_detection():
    """Render the live audio detection page"""
    with open("static/live.html", "r") as file:
        return file.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming audio data"""
    await websocket.accept()
    
    try:
        while True:
            # Receive audio chunks from client
            audio_data = await websocket.receive_bytes()
            
            # Convert to numpy array (assuming audio is sent as float32 values)
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            # Create a single segment from the received audio chunk
            segment = audio_np
            target_length = int(0.5 * 44100)  # 0.5 seconds at 44.1kHz
            
            # Ensure consistent length by zero-padding short segments
            if len(segment) < target_length:
                segment = np.pad(segment, (0, target_length - len(segment)), 'constant')
            elif len(segment) > target_length:
                segment = segment[:target_length]
            
            # Convert to mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=44100)
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), (128, 128))
            
            # Make prediction
            prediction = model.predict(np.array([mel_spectrogram]), verbose=0)
            predicted_label = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            # Determine class name and type
            if predicted_label < len(note_classes):
                class_name = note_classes[predicted_label]
                class_type = "note"
            else:
                class_name = chord_classes[predicted_label - len(note_classes)]
                class_type = "chord"
            
            # Send result back to client
            result = {
                "label": class_name,
                "type": class_type,
                "confidence": confidence
            }
            
            await websocket.send_json(result)
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(...), 
    segment_duration: float = 0.5, 
    auto_segment: bool = False,
    confidence_threshold: float = 0.2
):
    if not file.filename.endswith(('.mp3', '.wav', '.ogg')):
        raise HTTPException(400, detail="Invalid audio format. Please upload MP3, WAV, or OGG file.")
    
    if model is None:
        raise HTTPException(500, detail="Model not loaded. Please try again later.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        # Get audio duration
        y, sr = librosa.load(temp_file_path, sr=None, duration=10)  # Load just a bit to get duration
        audio_duration = librosa.get_duration(y=y, sr=sr)
        
        # Process audio file
        audio_segments, segment_times = load_and_segment_audio(
            temp_file_path, 
            segment_duration=segment_duration,
            auto_segment=auto_segment
        )
        
        mel_spectrograms = segments_to_mel_spectrograms(audio_segments)
        
        # Make predictions
        predictions = model.predict(mel_spectrograms)
        
        # Process prediction results
        predicted_labels = np.argmax(predictions, axis=1)
        
        results = []
        filtered_count = 0
        for i, label in enumerate(predicted_labels):
            confidence = float(np.max(predictions[i]))
            
            # Only include predictions that meet the confidence threshold
            if confidence >= confidence_threshold:
                if label < len(note_classes):
                    class_name = note_classes[label]
                    class_type = "note"
                else:
                    class_name = chord_classes[label - len(note_classes)]
                    class_type = "chord"
                
                results.append({
                    "time": float(segment_times[i]),
                    "label": str(class_name),
                    "type": class_type,
                    "confidence": confidence
                })
            else:
                filtered_count += 1
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return JSONResponse(content={
            "results": results,
            "filtered_count": filtered_count,
            "audio_duration": float(audio_duration)
        })
        
    except Exception as e:
        raise HTTPException(500, detail=f"Error processing audio: {str(e)}")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
