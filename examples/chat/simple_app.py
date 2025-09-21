#!/usr/bin/env python3
"""
Simplified VisionAgent demo app that works without the full vision-agent package.
This demonstrates the basic structure and can be extended when dependencies are available.
"""

import os
import base64
import tempfile
from typing import Any, Dict, List, Optional
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI(title="VisionAgent Demo", version="1.0.0")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str
    media: Optional[List[str]] = None

class Detection(BaseModel):
    label: str
    bbox: List[int]
    confidence: float
    mask: Optional[List[int]] = None

def process_image(image_data: bytes) -> Dict[str, Any]:
    """Process uploaded image and return basic information."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        # Get basic image information
        height, width, channels = image.shape
        
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Basic image analysis (placeholder for actual vision processing)
        analysis = {
            "dimensions": {"width": width, "height": height, "channels": channels},
            "file_size": len(image_data),
            "format": "image/jpeg",  # Assume JPEG for now
            "analysis": "This is a placeholder for actual vision analysis. In the full version, this would use VisionAgent to analyze the image content."
        }
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VisionAgent Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { 
                border: 2px dashed #ccc; 
                padding: 20px; 
                text-align: center; 
                margin: 20px 0;
                border-radius: 10px;
            }
            .upload-area:hover { border-color: #999; }
            button { 
                background: #007bff; 
                color: white; 
                padding: 10px 20px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer;
                margin: 10px;
            }
            button:hover { background: #0056b3; }
            .result { 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 20px 0;
                white-space: pre-wrap;
            }
            .error { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>VisionAgent Demo</h1>
            <p>This is a simplified demo of the VisionAgent application. Upload an image to see basic analysis.</p>
            
            <div class="upload-area">
                <h3>Upload an Image</h3>
                <input type="file" id="imageInput" accept="image/*" />
                <br><br>
                <button onclick="analyzeImage()">Analyze Image</button>
            </div>
            
            <div id="result"></div>
            
            <h3>Note</h3>
            <p>This is a simplified version that demonstrates the basic structure. The full VisionAgent application would include:</p>
            <ul>
                <li>Advanced computer vision analysis</li>
                <li>Object detection and recognition</li>
                <li>Code generation based on image content</li>
                <li>Integration with AI models (Anthropic, Google)</li>
            </ul>
        </div>

        <script>
            async function analyzeImage() {
                const fileInput = document.getElementById('imageInput');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {
                    resultDiv.innerHTML = '<div class="error">Please select an image first.</div>';
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    resultDiv.innerHTML = '<div>Analyzing image...</div>';
                    
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error('Analysis failed');
                    }
                    
                    const result = await response.json();
                    resultDiv.innerHTML = '<div class="result">' + JSON.stringify(result, null, 2) + '</div>';
                    
                } catch (error) {
                    resultDiv.innerHTML = '<div class="error">Error: ' + error.message + '</div>';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded image."""
    try:
        # Read image data
        image_data = await file.read()
        
        # Process the image
        analysis = process_image(image_data)
        
        return {
            "status": "success",
            "filename": file.filename,
            "analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat")
async def chat(message: Message):
    """Handle chat messages (placeholder for full VisionAgent functionality)."""
    return {
        "status": "success",
        "message": "This is a placeholder for VisionAgent chat functionality. In the full version, this would process your message and generate appropriate responses using AI models.",
        "original_message": message.content
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "VisionAgent Demo is running"}

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT_BACKEND", "8000"))
    
    print(f"Starting VisionAgent Demo on port {port}")
    print(f"Open your browser to: http://localhost:{port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
