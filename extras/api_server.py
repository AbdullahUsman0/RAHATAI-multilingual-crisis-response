"""
FastAPI Server for RahatAI
Provides REST API endpoints for Android app integration (similar to CrisisConnect)
Supports: Classification, NER, RAG, Speech-to-Text (Whisper), and Roman Urdu
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import numpy as np
import io
import base64

# Initialize FastAPI app
app = FastAPI(
    title="RahatAI API",
    description="Multilingual Crisis Response NLP API - Supports English, Urdu, and Roman Urdu",
    version="1.0.0"
)

# CORS middleware for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Android app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load label mappings
with open("Data/Preprocessed/label_mappings.json", "r") as f:
    label_mappings = json.load(f)
    # Create mapping that supports both int and string keys
    idx_to_label = {}
    for k, v in label_mappings["idx_to_label"].items():
        idx_to_label[int(k)] = v  # Support int keys
        idx_to_label[k] = v  # Support string keys

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class TextRequest(BaseModel):
    text: str
    model: Optional[str] = "transformer"  # transformer, svm, naive_bayes

class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    probabilities: dict

class NERResponse(BaseModel):
    locations: List[str]
    phone_numbers: List[str]
    persons: List[str]
    organizations: List[str]
    resources: List[str]

class RAGRequest(BaseModel):
    question: str
    use_voice: Optional[bool] = False

class RAGResponse(BaseModel):
    answer: str
    sources: List[dict]
    transcribed_text: Optional[str] = None

# ============================================================
# LOAD MODELS (Lazy loading)
# ============================================================

_models_cache = {}

def get_classifier(model_name: str = "transformer"):
    """Get classifier model (cached)"""
    if model_name not in _models_cache:
        if model_name == "transformer":
            from Scripts.classification.transformer_models import TransformerClassifier
            model = TransformerClassifier()
            model.load("Models/transformer")
        elif model_name == "svm":
            from Scripts.classification.ml_models import SVMClassifier
            model = SVMClassifier()
            model.load("Models/svm.pkl")
        elif model_name == "naive_bayes":
            from Scripts.classification.ml_models import NaiveBayesClassifier
            model = NaiveBayesClassifier()
            model.load("Models/naive_bayes.pkl")
        else:
            raise ValueError(f"Unknown model: {model_name}")
        _models_cache[model_name] = model
    return _models_cache[model_name]

def get_ner_model():
    """Get NER model (cached)"""
    if "ner" not in _models_cache:
        from Scripts.ner.ner_extractor import MultilingualNER
        model = MultilingualNER()
        _models_cache["ner"] = model
    return _models_cache["ner"]

def get_rag_system():
    """Get RAG system (cached)"""
    if "rag" not in _models_cache:
        from Scripts.rag.query_rag import RAGSystem
        rag = RAGSystem(use_rag=True)
        _models_cache["rag"] = rag
    return _models_cache["rag"]

def get_whisper_transcriber():
    """Get Whisper transcriber (cached)"""
    if "whisper" not in _models_cache:
        try:
            from Scripts.speech.whisper_transcriber import WhisperTranscriber
            transcriber = WhisperTranscriber(model_size="base")
            _models_cache["whisper"] = transcriber
        except ImportError:
            _models_cache["whisper"] = None
    return _models_cache["whisper"]

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "RahatAI API",
        "version": "1.0.0",
        "description": "Multilingual Crisis Response NLP API",
        "endpoints": {
            "/classify": "POST - Classify crisis text",
            "/ner": "POST - Extract named entities",
            "/rag": "POST - RAG query (text or voice)",
            "/transcribe": "POST - Speech-to-text (Whisper)",
            "/health": "GET - Health check"
        },
        "supported_languages": ["English", "Urdu", "Roman-Urdu"],
        "supported_models": ["transformer", "svm", "naive_bayes"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RahatAI API"}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: TextRequest):
    """
    Classify crisis text into categories
    
    Supports: English, Urdu, Roman-Urdu
    """
    try:
        model = get_classifier(request.model)
        
        # Predict
        prediction = int(model.predict([request.text])[0])  # Convert to int
        probabilities = model.predict_proba([request.text])[0]
        
        # Format response
        prob_dict = {
            idx_to_label[str(i)]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return ClassificationResponse(
            category=idx_to_label[str(prediction)],
            confidence=float(probabilities[prediction]),
            probabilities=prob_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/ner", response_model=NERResponse)
async def extract_entities(request: TextRequest):
    """
    Extract named entities from text
    
    Extracts: Locations, Phone Numbers, Persons, Organizations, Resources
    Supports: English, Urdu, Roman-Urdu
    """
    try:
        ner = get_ner_model()
        entities = ner.extract_all(request.text)
        
        return NERResponse(
            locations=entities.get("locations", []),
            phone_numbers=entities.get("phone_numbers", []),
            persons=entities.get("persons", []),
            organizations=entities.get("organizations", []),
            resources=entities.get("resources", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER error: {str(e)}")

@app.post("/rag", response_model=RAGResponse)
async def rag_query(
    question: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    RAG Query - Answer questions using document retrieval
    
    Can accept:
    - Text question (via 'question' form field)
    - Voice question (via 'audio' file upload - will be transcribed with Whisper)
    
    Supports: English, Urdu, Roman-Urdu (including Roman Urdu from voice)
    """
    try:
        rag = get_rag_system()
        transcribed_text = None
        
        # Handle voice input
        if audio is not None:
            transcriber = get_whisper_transcriber()
            if transcriber is None:
                raise HTTPException(
                    status_code=500, 
                    detail="Whisper not available. Install with: pip install openai-whisper"
                )
            
            # Read audio file
            audio_bytes = await audio.read()
            
            # Transcribe
            transcribed_text = transcriber.transcribe_for_rag(audio_bytes)
            question = transcribed_text
        
        if not question:
            raise HTTPException(status_code=400, detail="No question provided (text or audio)")
        
        # Query RAG system
        result = rag.query(question)
        
        return RAGResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            transcribed_text=transcribed_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), language: Optional[str] = Form(None)):
    """
    Transcribe audio to text using Whisper
    
    Supports: English, Urdu, Roman-Urdu
    """
    try:
        transcriber = get_whisper_transcriber()
        if transcriber is None:
            raise HTTPException(
                status_code=500,
                detail="Whisper not available. Install with: pip install openai-whisper"
            )
        
        # Read audio file
        audio_bytes = await audio.read()
        
        # Transcribe
        result = transcriber.transcribe(audio_bytes, language=language)
        
        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "")
                }
                for seg in result.get("segments", [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/classify-voice")
async def classify_voice(
    audio: UploadFile = File(...),
    model: Optional[str] = Form("transformer")
):
    """
    Classify crisis text from voice input
    
    1. Transcribes audio using Whisper
    2. Classifies the transcribed text
    
    Supports: English, Urdu, Roman-Urdu voice input
    """
    try:
        # Step 1: Transcribe
        transcriber = get_whisper_transcriber()
        if transcriber is None:
            raise HTTPException(
                status_code=500,
                detail="Whisper not available. Install with: pip install openai-whisper"
            )
        
        audio_bytes = await audio.read()
        transcription = transcriber.transcribe(audio_bytes)
        text = transcription["text"]
        
        # Step 2: Classify
        classifier = get_classifier(model)
        prediction = classifier.predict([text])[0]
        probabilities = classifier.predict_proba([text])[0]
        
        prob_dict = {
            idx_to_label[str(i)]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "transcribed_text": text,
            "category": idx_to_label[str(prediction)],
            "confidence": float(probabilities[prediction]),
            "probabilities": prob_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice classification error: {str(e)}")

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Starting RahatAI API Server")
    print("=" * 70)
    print("\nAvailable endpoints:")
    print("  POST /classify - Classify crisis text")
    print("  POST /ner - Extract named entities")
    print("  POST /rag - RAG query (text or voice)")
    print("  POST /transcribe - Speech-to-text")
    print("  POST /classify-voice - Classify from voice")
    print("  GET /health - Health check")
    print("\nServer will run on: http://localhost:8080")
    print("For Android emulator: http://10.0.2.2:8080")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8080)

