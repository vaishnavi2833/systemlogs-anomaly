import os
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from process.project_parser import parse_logs  # Ensure this is correctly imported
from logcnn import LogCNN

app = FastAPI()

# Folder paths
RAW_LOG_DIR = "uploaded_logs"
PARSED_LOG_DIR = "parsed_logs"
PROCESSED_DATA_DIR = "processed_data/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "log_cnn_model.pt"

# Ensure directories exist
os.makedirs(RAW_LOG_DIR, exist_ok=True)
os.makedirs(PARSED_LOG_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load model at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        print("✅ Attempting to load model...")
        
        # Check if the model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found at {MODEL_PATH}")
            model = None
            return
        
        # Loading the model
        model = LogCNN(num_classes=2)  # Replace 2 with your actual number of classes
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        model = model.to(DEVICE)
        model.eval()
        print("✅ Model loaded successfully.")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None


def pad_and_reshape(x):
    # Apply padding and reshaping, adjusting the input size for the model
    return torch.unsqueeze(F.pad(torch.tensor(x, dtype=torch.float32), pad=(1, 1, 1, 1)), dim=1)


@app.post("/upload-log/")
async def upload_log(file: UploadFile = File(...)):
    try:
        # Ensure the model is loaded
        print("in try block")
        print(model)
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")

        # Save the uploaded file
        raw_log_path = os.path.join(RAW_LOG_DIR, file.filename)
        with open(raw_log_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse and process the uploaded log file
        # parse_logs(input_path=raw_log_path, output_dir=PARSED_LOG_DIR, save_processed_dir=PROCESSED_DATA_DIR)

        # Load processed test data (X_test) and original lines
        X_test_path = os.path.join(PROCESSED_DATA_DIR, "x_test_tf-idf_v5.npy")
        lines_path = os.path.join(PROCESSED_DATA_DIR, "original_lines.txt")

        # Check if required processed files exist
        if not os.path.exists(X_test_path):
            raise HTTPException(status_code=500, detail="Processed data not found.")
        if not os.path.exists(lines_path):
            raise HTTPException(status_code=500, detail="Original lines not found.")

        # Load test data and original log lines
        X_test = np.load(X_test_path)
        with open(lines_path, "r") as f:
            original_lines = f.read().splitlines()

        # Reshape the input tensor before feeding to the model
        input_tensor = pad_and_reshape(X_test).to(DEVICE)

        # Predict anomalies using the model
        with torch.no_grad():
            _, probas = model(input_tensor)
            predictions = torch.argmax(probas, dim=1).cpu().numpy()

        # Extract anomalous lines based on predictions (assuming 1 is anomalous)
        anomalous_lines = [line for line, pred in zip(original_lines, predictions) if pred == 1]

        return JSONResponse(content={
            "message": "✅ Log processed and analyzed",
            "anomalies_detected": len(anomalous_lines),
            "anomalous_lines": anomalous_lines
        })

    except HTTPException as http_exc:
        return JSONResponse(content={"error": str(http_exc.detail)}, status_code=http_exc.status_code)
    except Exception as e:
        return JSONResponse(content={"error": f"Exception occurred: {str(e)}"}, status_code=500)


@app.get("/")
async def root():
    return {"message": "Welcome to the Log Anomaly Detection API 🚀"}
