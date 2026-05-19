from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Import the processing function we just refactored
from structured_extraction import process_image

app = FastAPI(title="Accounting OCR Extractor")

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure templates directory exists
if not os.path.exists("templates"):
    os.makedirs("templates")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount structured_output so we can serve the debug image directly to the UI
if not os.path.exists("structured_output"):
    os.makedirs("structured_output")
app.mount("/output", StaticFiles(directory="structured_output"), name="output")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/extract")
async def extract(file: UploadFile = File(...), date: str = Form(None)):
    try:
        contents = await file.read()
        # Process the image bytes through our OCR pipeline
        structured_data = process_image(contents, date_val=date)
        
        return JSONResponse(content={
            "success": True, 
            "data": structured_data, 
            "debug_image": "/output/label_matching_debug.jpg"
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    print("Starting FastAPI server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
