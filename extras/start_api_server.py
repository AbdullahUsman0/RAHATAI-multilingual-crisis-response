"""
Start API Server with Model Pre-loading
This pre-loads models to avoid first-request delays
"""
import sys
import warnings
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("RAHATAI API SERVER - Starting with Model Pre-loading")
print("=" * 70)

# Pre-load models
print("\nPre-loading models (this may take a minute)...")

try:
    print("  Loading Transformer model...")
    from Scripts.classification.transformer_models import TransformerClassifier
    transformer = TransformerClassifier()
    try:
        transformer.load("Models/transformer")
        print("  ✅ Transformer loaded")
    except Exception as load_err:
        print(f"  ⚠️  Could not load transformer: {load_err}")
        print("  ℹ️  Transformer will be initialized on first use")
except Exception as e:
    print(f"  ⚠️  Transformer: {e}")

try:
    print("  Loading SVM model...")
    from Scripts.classification.ml_models import SVMClassifier
    svm = SVMClassifier()
    svm.load("Models/svm.pkl")
    print("  ✅ SVM loaded")
except Exception as e:
    print(f"  ⚠️  SVM: {e}")

try:
    print("  Loading Naive Bayes model...")
    from Scripts.classification.ml_models import NaiveBayesClassifier
    nb = NaiveBayesClassifier()
    nb.load("Models/naive_bayes.pkl")
    print("  ✅ Naive Bayes loaded")
except Exception as e:
    print(f"  ⚠️  Naive Bayes: {e}")

try:
    print("  Loading NER model...")
    from Scripts.ner.ner_extractor import MultilingualNER
    ner = MultilingualNER()
    print("  ✅ NER loaded")
except Exception as e:
    print(f"  ⚠️  NER: {e}")

try:
    print("  Loading Whisper...")
    from Scripts.speech.whisper_transcriber import WhisperTranscriber
    whisper = WhisperTranscriber(model_size="base")
    print("  ✅ Whisper loaded")
except Exception as e:
    print(f"  ⚠️  Whisper: {e}")

try:
    print("  Loading RAG system...")
    from Scripts.rag.query_rag import RAGSystem
    rag = RAGSystem(use_rag=True)
    print("  ✅ RAG loaded")
except Exception as e:
    print(f"  ⚠️  RAG: {e} (may need vectorstore setup)")

print("\n" + "=" * 70)
print("Starting API server...")

# Check if port 8080 is in use and stop it
import subprocess
try:
    result = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True,
        text=True
    )
    lines = result.stdout.split('\n')
    pid = None
    for line in lines:
        if ':8080' in line and 'LISTENING' in line:
            parts = line.split()
            if len(parts) >= 5:
                pid = parts[-1]
                break
    
    if pid:
        print(f"  Stopping existing server on port 8080 (PID: {pid})...")
        try:
            subprocess.run(["taskkill", "/F", "/PID", pid], 
                         capture_output=True, check=True)
            import time
            time.sleep(1)  # Wait for port to be released
            print("  ✅ Port 8080 is now free")
        except:
            print("  ⚠️  Could not stop existing server. Please stop it manually.")
except:
    pass  # If netstat fails, continue anyway

print("Server will be available at:")
print("  - Local: http://localhost:8080")
print("  - Android Emulator: http://10.0.2.2:8080")
print("=" * 70)
print()

# Now start the server
from api_server import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8080)

