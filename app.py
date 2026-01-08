"""
RahatAI - Simple Demo App
Multilingual Crisis Response NLP System
Dark Theme Redesign
"""

import streamlit as st
import sys
from pathlib import Path
import json
import pickle
import os

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# FFmpeg should be in system PATH or configured via FFMPEG_PATH environment variable
ffmpeg_path = os.environ.get("FFMPEG_PATH")
if ffmpeg_path and os.path.exists(ffmpeg_path) and ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + ffmpeg_path

# ============================================================
# DARK THEME CSS
# ============================================================
st.markdown("""
<style>
    /* Main dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar dark theme */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #2d2d44;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #4fc3f7 !important;
        text-shadow: 0 0 10px rgba(79, 195, 247, 0.3);
    }
    
    /* Text */
    .stMarkdown, p, li, span {
        color: #e0e0e0 !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #2d2d44 !important;
        color: #e0e0e0 !important;
        border: 1px solid #3d3d5c !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border: 1px solid #4fc3f7 !important;
        box-shadow: 0 0 5px rgba(79, 195, 247, 0.3) !important;
    }
    
    /* Selectbox and radio */
    .stSelectbox > div > div,
    .stRadio > div {
        background-color: #2d2d44 !important;
        color: #e0e0e0 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Success/Info/Error boxes */
    .stSuccess {
        background-color: #1b5e20 !important;
        border-left: 4px solid #4caf50 !important;
        color: #c8e6c9 !important;
    }
    
    .stInfo {
        background-color: #0d47a1 !important;
        border-left: 4px solid #2196f3 !important;
        color: #bbdefb !important;
    }
    
    .stWarning {
        background-color: #e65100 !important;
        border-left: 4px solid #ff9800 !important;
        color: #ffe0b2 !important;
    }
    
    .stError {
        background-color: #b71c1c !important;
        border-left: 4px solid #f44336 !important;
        color: #ffcdd2 !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1e1e2e !important;
        border: 1px solid #3d3d5c !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d2d44 !important;
        color: #e0e0e0 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #4fc3f7 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0bec5 !important;
    }
    
    /* JSON viewer */
    .stJson {
        background-color: #1e1e2e !important;
        border: 1px solid #3d3d5c !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* JSON viewer text */
    .stJson pre {
        background-color: #1e1e2e !important;
        color: #e0e0e0 !important;
        border: none !important;
    }
    
    /* Custom probability cards */
    .prob-card {
        background: linear-gradient(135deg, #2d2d44 0%, #1a1a2e 100%);
        border: 1px solid #3d3d5c;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .prob-card:hover {
        border-color: #4fc3f7;
        box-shadow: 0 0 10px rgba(79, 195, 247, 0.2);
    }
    
    .prob-label {
        color: #b0bec5;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    .prob-value {
        color: #4fc3f7;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .prob-bar-container {
        background-color: #1a1a2e;
        border-radius: 4px;
        height: 8px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    
    .prob-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: #2d2d44 !important;
        border: 1px solid #3d3d5c !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #4fc3f7 !important;
    }
    
    /* Divider */
    hr {
        border-color: #3d3d5c !important;
    }
    
    /* Caption */
    .stCaption {
        color: #90a4ae !important;
    }
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title="RahatAI - Crisis Response NLP",
    page_icon="🆘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title with gradient effect
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-size: 3rem;
               margin-bottom: 0.5rem;
               font-weight: 700;">🆘 RahatAI</h1>
    <p style="color: #b0bec5; font-size: 1.2rem; margin-top: 0;">Multilingual Crisis Response & Disaster Management NLP System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar - Model Selection
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("---")

task = st.sidebar.selectbox(
    "Choose Task:",
    ["Classification", "NER", "Summarization", "Misinformation Detection", "RAG Query", "About"]
)

# Load label mappings
@st.cache_data
def load_label_mappings():
    with open("Data/Preprocessed/label_mappings.json", "r") as f:
        return json.load(f)

label_mappings = load_label_mappings()
idx_to_label = label_mappings['idx_to_label']

# ============================================================
# TASK 1: CLASSIFICATION
# ============================================================
if task == "Classification":
    st.header("📋 Crisis Text Classification")
    st.markdown("Classify crisis-related text into 6 categories using any of the 5 available models")
    
    # Check which models are available
    from pathlib import Path
    available_models = []
    model_options = []
    
    # Check each model
    if Path("Models/transformer").exists():
        available_models.append("Transformer")
        model_options.append("Transformer (Best - 73.35%)")
    
    if Path("Models/svm.pkl").exists():
        available_models.append("SVM")
        model_options.append("SVM (Fast - 66.53%)")
    
    if Path("Models/naive_bayes.pkl").exists():
        available_models.append("Naive Bayes")
        model_options.append("Naive Bayes (Baseline - 48.76%)")
    
    if Path("Models/lstm").exists():
        available_models.append("LSTM")
        model_options.append("LSTM (Deep Learning - ~60%) ")
    
    if Path("Models/cnn").exists():
        available_models.append("CNN")
        model_options.append("CNN (Deep Learning - ~52%) ")
    
    # If no models available, show warning
    if not model_options:
        st.error("❌ No trained models found. Please train at least one model first.")
        st.info("Start with: `python RunScripts/train_svm.py`")
        st.stop()
    
    # Model selection - Only show available models
    model_choice = st.selectbox(
        "Select Model:",
        model_options
    )
    
    
    # Input mode selection
    input_mode = st.radio(
        "Input Mode:",
        ["Text", "🎤 Record Voice", " Upload Audio File"],
        horizontal=True
    )
    
    user_text = ""
    
    if input_mode == "Text":
        user_text = st.text_area(
            "Enter crisis text:",
            placeholder="Example: Emergency in Karachi. Need food and water. Contact 0300-1234567.",
            height=100
        )
    elif input_mode == "🎤 Record Voice":
        st.info("🎤 Click the button below to record your voice directly in the browser")
        try:
            from audio_recorder_streamlit import audio_recorder
            
            # Record audio
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e8b4c8",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x",
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                # Transcribe with Whisper - pass bytes directly (Whisper handles temp file internally)
                with st.spinner("🎤 Transcribing audio with Whisper..."):
                    from Scripts.speech.whisper_transcriber import WhisperTranscriber
                    whisper_model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
                    transcriber = WhisperTranscriber(model_size=whisper_model_size)
                    result = transcriber.transcribe(audio_bytes)  # Pass bytes directly
                    user_text = result["text"] if isinstance(result, dict) else str(result)
                
                # Show transcribed text
                st.success("✅ Transcription Complete!")
                st.write("**Transcribed Text:**")
                st.info(user_text)
                
        except ImportError:
            st.error("❌ Audio recorder not installed")
            st.info("Install it with: `pip install audio-recorder-streamlit`")
            st.info("Or use 'Upload Audio File' option instead")
        except FileNotFoundError as e:
            error_msg = str(e).lower()
            if 'ffmpeg' in error_msg:
                st.error("❌ FFmpeg is not installed!")
                st.warning("Whisper requires FFmpeg to process audio files.")
                st.info("**Install FFmpeg:** See `INSTALL_FFMPEG.md` for instructions")
                st.markdown("**Quick install:** `choco install ffmpeg` (requires Chocolatey)")
            else:
                st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Error recording/transcribing audio: {e}")
            error_msg = str(e).lower()
            if 'ffmpeg' in error_msg or 'winerror 2' in error_msg:
                st.info("💡 This might be an FFmpeg issue. See `INSTALL_FFMPEG.md` for installation instructions.")
            else:
                st.info("Make sure Whisper is installed: pip install openai-whisper")
            user_text = ""
    else:  # Upload Audio File
        st.info("📁 Upload an audio file (MP3, WAV, M4A, etc.) - Supports English, Urdu, and Roman Urdu")
        audio_file = st.file_uploader(
            "Upload Audio File:",
            type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm'],
            help="Record or upload audio in English, Urdu, or Roman Urdu"
        )
        
        if audio_file:
            try:
                # Save uploaded file temporarily
                import tempfile
                import os
                
                # Create temp file and ensure it's properly saved
                file_ext = os.path.splitext(audio_file.name)[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, mode='wb') as tmp_file:
                    tmp_file.write(audio_file.read())
                    tmp_file.flush()
                    tmp_path = tmp_file.name
                
                # Ensure file exists and is readable
                if not os.path.exists(tmp_path):
                    raise FileNotFoundError(f"Temporary file not created: {tmp_path}")
                
                try:
                        # Transcribe with Whisper
                        with st.spinner("🎤 Transcribing audio with Whisper..."):
                            from Scripts.speech.whisper_transcriber import WhisperTranscriber
                            whisper_model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
                            transcriber = WhisperTranscriber(model_size=whisper_model_size)
                            result = transcriber.transcribe(tmp_path)
                        user_text = result["text"] if isinstance(result, dict) else str(result)
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                
                # Show transcribed text
                st.success("✅ Transcription Complete!")
                st.write("**Transcribed Text:**")
                st.info(user_text)
                
            except FileNotFoundError as e:
                error_msg = str(e).lower()
                if 'ffmpeg' in error_msg:
                    st.error("❌ FFmpeg is not installed!")
                    st.warning("Whisper requires FFmpeg to process audio files.")
                    st.info("**Install FFmpeg:** See `INSTALL_FFMPEG.md` for instructions")
                    st.markdown("**Quick install:** `choco install ffmpeg` (requires Chocolatey)")
                else:
                    st.error(f"Error: {e}")
                user_text = ""
            except Exception as e:
                st.error(f"Error transcribing audio: {e}")
                error_msg = str(e).lower()
                if 'ffmpeg' in error_msg or 'winerror 2' in error_msg:
                    st.info("💡 This might be an FFmpeg issue. See `INSTALL_FFMPEG.md` for installation instructions.")
                else:
                    st.info("Make sure Whisper is installed: pip install openai-whisper")
                user_text = ""
    
    if st.button("Classify", type="primary"):
        if user_text.strip():
            try:
                # Load model based on choice
                with st.spinner("Loading model..."):
                    if "Transformer" in model_choice:
                        from Scripts.classification.transformer_models import TransformerClassifier
                        model = TransformerClassifier()
                        model.load("Models/transformer")
                        model_name = "Transformer"
                    elif "SVM" in model_choice:
                        from Scripts.classification.ml_models import SVMClassifier
                        model = SVMClassifier()
                        model.load("Models/svm.pkl")
                        model_name = "SVM"
                    elif "Naive Bayes" in model_choice:
                        from Scripts.classification.ml_models import NaiveBayesClassifier
                        model = NaiveBayesClassifier()
                        model.load("Models/naive_bayes.pkl")
                        model_name = "Naive Bayes"
                    elif "LSTM" in model_choice:
                        from Scripts.classification.dl_models import LSTMClassifier
                        from pathlib import Path
                        lstm_path = Path("Models/lstm")
                        if not lstm_path.exists():
                            st.error("❌ LSTM model not found. Please train it first:")
                            st.code("python RunScripts/train_lstm_efficient.py", language="bash")
                            st.stop()
                        try:
                            model = LSTMClassifier()
                            model.load("Models/lstm")
                            model_name = "LSTM"
                        except Exception as e:
                            error_msg = str(e)
                            if "batch_shape" in error_msg or "TensorFlow" in error_msg or "version" in error_msg.lower():
                                st.error("❌ TensorFlow version compatibility issue with LSTM model")
                                st.warning("The LSTM model was saved with a different TensorFlow version.")
                                st.info("**Solutions:**")
                                st.markdown("""
                                1. **Retrain the model** (Recommended):
                                   ```bash
                                   python RunScripts/train_lstm_efficient.py
                                   ```
                                2. **Or install compatible TensorFlow version**:
                                   ```bash
                                   pip install tensorflow-cpu==2.10.0 --force-reinstall
                                   ```
                                """)
                                st.info("💡 For now, you can use other models (SVM, Transformer, Naive Bayes) which work fine.")
                            else:
                                st.error(f"❌ Error loading LSTM model: {error_msg}")
                            st.stop()
                    elif "CNN" in model_choice:
                        from Scripts.classification.dl_models import CNNClassifier
                        from pathlib import Path
                        cnn_path = Path("Models/cnn")
                        if not cnn_path.exists():
                            st.error("❌ CNN model not found. Please train it first:")
                            st.code("python RunScripts/train_cnn.py", language="bash")
                            st.stop()
                        try:
                            model = CNNClassifier()
                            model.load("Models/cnn")
                            model_name = "CNN"
                        except Exception as e:
                            error_msg = str(e)
                            if "batch_shape" in error_msg or "TensorFlow" in error_msg or "version" in error_msg.lower():
                                st.error("❌ TensorFlow version compatibility issue with CNN model")
                                st.warning("The CNN model was saved with a different TensorFlow version.")
                                st.info("**Solutions:**")
                                st.markdown("""
                                1. **Retrain the model** (Recommended):
                                   ```bash
                                   python RunScripts/train_cnn.py
                                   ```
                                2. **Or install compatible TensorFlow version**:
                                   ```bash
                                   pip install tensorflow-cpu==2.10.0 --force-reinstall
                                   ```
                                """)
                                st.info("💡 For now, you can use other models (SVM, Transformer, Naive Bayes) which work fine.")
                            else:
                                st.error(f"❌ Error loading CNN model: {error_msg}")
                            st.stop()
                
                # Predict
                with st.spinner("Classifying..."):
                    prediction = int(model.predict([user_text])[0])
                    probabilities = model.predict_proba([user_text])[0]
                
                # Show results
                st.success(f"✅ Classification Complete! (Model: {model_name})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Category", idx_to_label[str(prediction)])
                with col2:
                    st.metric("Confidence", f"{probabilities[prediction]*100:.1f}%")
                
                # Show probabilities with better styling
                st.subheader("📊 All Category Probabilities")
                
                # Sort by probability (descending)
                sorted_probs = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
                
                # Create a custom HTML display for probabilities
                prob_html = """
                <div style="background: linear-gradient(135deg, #2d2d44 0%, #1a1a2e 100%); 
                            border: 1px solid #3d3d5c; 
                            border-radius: 8px; 
                            padding: 1.5rem; 
                            margin-top: 1rem;">
                """
                
                for idx, prob in sorted_probs:
                    label = idx_to_label[str(idx)]
                    percentage = prob * 100
                    bar_width = percentage
                    is_predicted = idx == prediction
                    
                    # Highlight the predicted category
                    border_style = "2px solid #4fc3f7" if is_predicted else "1px solid #3d3d5c"
                    bg_style = "rgba(79, 195, 247, 0.1)" if is_predicted else "transparent"
                    
                    prob_html += f"""
                    <div style="margin-bottom: 1rem; 
                                padding: 0.75rem; 
                                background: {bg_style}; 
                                border: {border_style}; 
                                border-radius: 6px;
                                transition: all 0.3s ease;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="color: #e0e0e0; font-weight: 500;">{label}</span>
                            <span style="color: #4fc3f7; font-weight: 600; font-size: 1.1rem;">{percentage:.1f}%</span>
                        </div>
                        <div style="background-color: #1a1a2e; border-radius: 4px; height: 10px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                        height: 100%; 
                                        width: {bar_width}%; 
                                        border-radius: 4px; 
                                        transition: width 0.5s ease;"></div>
                        </div>
                    </div>
                    """
                
                prob_html += "</div>"
                st.markdown(prob_html, unsafe_allow_html=True)
                
            except FileNotFoundError as e:
                st.error(f"Model file not found: {e}")
                st.info(f"Please train the {model_name} model first. Check RunScripts/ folder for training scripts.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure the model is trained. Check RunScripts/ folder for training scripts.")
        else:
            st.warning("⚠️ Please enter some text")

# ============================================================
# TASK 2: NER (Named Entity Recognition)
# ============================================================
elif task == "NER":
    st.header("🏷 Named Entity Recognition")
    st.markdown("Extract locations, phone numbers, resources, persons, and organizations from text")
    
    # Input
    user_text = st.text_area(
        "Enter crisis text:",
        placeholder="Example: Emergency in Lahore. Contact Dr. Ahmed at 0300-1234567. Need food and medical supplies.",
        height=100
    )
    
    if st.button("Extract Entities", type="primary"):
        if user_text.strip():
            try:
                # Load NER
                with st.spinner("Loading NER model..."):
                    from Scripts.ner.ner_extractor import MultilingualNER
                    ner = MultilingualNER()
                
                # Extract
                with st.spinner("Extracting entities..."):
                    entities = ner.extract_all(user_text)
                
                # Show results
                st.success("✅ Extraction Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📍 Locations")
                    if entities['locations']:
                        for loc in entities['locations']:
                            st.markdown(f"• {loc}")
                    else:
                        st.info("No locations found")
                    
                    st.subheader("📞 Phone Numbers")
                    if entities['phone_numbers']:
                        for phone in entities['phone_numbers']:
                            st.markdown(f"• {phone}")
                    else:
                        st.info("No phone numbers found")
                    
                    st.subheader("📦 Resources")
                    if entities['resources']:
                        for res in entities['resources']:
                            st.markdown(f"• {res}")
                    else:
                        st.info("No resources found")
                
                with col2:
                    st.subheader("👤 Persons")
                    if entities['persons']:
                        for person in entities['persons']:
                            st.markdown(f"• {person}")
                    else:
                        st.info("No persons found")
                    
                    st.subheader("🏢 Organizations")
                    if entities['organizations']:
                        for org in entities['organizations']:
                            st.markdown(f"• {org}")
                    else:
                        st.info("No organizations found")
                
                # Show all entities for debugging
                with st.expander("🔍 View All Raw Entities"):
                    st.json(entities.get('all_entities', []))
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Please enter some text")

# ============================================================
# TASK 3: SUMMARIZATION
# ============================================================
elif task == "Summarization":
    st.header("📝 Text Summarization")
    st.markdown("Generate concise summaries of crisis reports using BART model")
    
    # Explanation
    with st.expander("ℹ️ How Summarization Works"):
        st.markdown("""
        **Model Used:** Facebook BART (Bidirectional and Auto-Regressive Transformers)
        
        **How it works:**
        1. **Input Processing**: Text is tokenized and truncated to 1024 tokens if too long
        2. **Abstractive Summarization**: BART generates a new summary (not just extracts sentences)
        3. **Length Control**: You can set min/max length for the summary
        4. **Output**: Returns a concise, coherent summary of the input text
        
        **Best for:**
        - Long crisis reports
        - Multiple related incidents
        - News articles about disasters
        - Emergency situation updates
        
        **Limitations:**
        - Works best with English text
        - May lose some specific details in very long texts
        - Requires at least 50 characters of input
        """)
    
    # Input
    user_text = st.text_area(
        "Enter text to summarize:",
        placeholder="Paste long crisis report here...",
        height=200
    )
    
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum summary length:", 50, 300, 150)
    with col2:
        min_length = st.slider("Minimum summary length:", 20, 100, 30)
    
    if st.button("Summarize", type="primary"):
        if user_text.strip():
            if len(user_text) < 50:
                st.warning("⚠️ Text too short to summarize. Add more content (minimum 50 characters).")
            else:
                try:
                    # Load summarizer
                    with st.spinner("Loading BART summarization model..."):
                        from Scripts.summarization.summarizer import CrisisSummarizer
                        summarizer = CrisisSummarizer()
                    
                    # Summarize
                    with st.spinner("Generating summary..."):
                        summary = summarizer.summarize(user_text, max_length=max_length, min_length=min_length)
                    
                    # Show results
                    st.success("✅ Summarization Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Length", f"{len(user_text)} characters")
                    with col2:
                        st.metric("Summary Length", f"{len(summary)} characters")
                    
                    st.subheader("📄 Summary")
                    st.info(summary)
                    
                    # Show compression ratio
                    compression = (1 - len(summary) / len(user_text)) * 100
                    st.caption(f"Compression: {compression:.1f}% reduction")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Make sure transformers library is installed: pip install transformers")
        else:
            st.warning("⚠️ Please enter some text")

# ============================================================
# TASK 4: MISINFORMATION DETECTION
# ============================================================
elif task == "Misinformation Detection":
    st.header("🔍 Misinformation Detection")
    st.markdown("Detect if crisis information is verified or potentially false")
    
    # Input
    user_text = st.text_area(
        "Enter crisis text to verify:",
        placeholder="Example: Rumor says cyclone approaching. Unconfirmed reports.",
        height=100
    )
    
    if st.button("Detect", type="primary"):
        if user_text.strip():
            try:
                # Load detector
                with st.spinner("Loading detection model..."):
                    from Scripts.misinformation.detector import MisinformationDetector
                    detector = MisinformationDetector()
                
                # Predict
                with st.spinner("Analyzing..."):
                    prediction = detector.predict([user_text])[0]
                    probabilities = detector.predict_proba([user_text])[0]
                
                # Show results
                st.success("✅ Detection Complete!")
                
                if prediction == 1:
                    st.error(f"⚠️ Potential Misinformation (Confidence: {probabilities[1]*100:.1f}%)")
                else:
                    st.success(f"✓ Likely Verified (Confidence: {probabilities[0]*100:.1f}%)")
                
                # Show linguistic features
                features = detector.extract_linguistic_features(user_text)
                
                with st.expander("View Analysis Details"):
                    st.write("**Linguistic Features:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Uncertainty Markers", features['uncertainty_count'])
                        st.metric("Exclamation Marks", features['exclamation_count'])
                    with col2:
                        st.metric("Credibility Markers", features['credibility_count'])
                        st.metric("Question Marks", features['question_count'])
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Please enter some text")

# ============================================================
# TASK 5: RAG QUERY
# ============================================================
elif task == "RAG Query":
    st.header("🔍 RAG Query System")
    st.markdown("Ask questions about disaster response procedures and get verified answers from official documents")
    
    # Check if RAG is set up
    vectorstore_path = Path("Models/rag_vectorstore")
    if not vectorstore_path.exists():
        st.warning("⚠️ RAG vectorstore not found. Please set it up first:")
        st.code("python RunScripts/SETUP_RAG_WITH_DOCUMENTS.py", language="bash")
    else:
        # Input mode selection
        input_mode = st.radio(
            "Input Mode:",
            ["Text", "🎤 Record Voice", "📁 Upload Audio File"],
            horizontal=True
        )
        
        user_question = ""
        
        if input_mode == "Text":
            user_question = st.text_input(
                "Enter your question:",
                placeholder="Example: What is the emergency helpline number?",
            )
        elif input_mode == "🎤 Record Voice":
            st.info("🎤 Click the button below to record your question directly in the browser")
            try:
                from audio_recorder_streamlit import audio_recorder
                
                # Record audio
                audio_bytes = audio_recorder(
                    text="Click to record",
                    recording_color="#e8b4c8",
                    neutral_color="#6aa36f",
                    icon_name="microphone",
                    icon_size="2x",
                )
                
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Transcribe with Whisper - pass bytes directly (Whisper handles temp file internally)
                    with st.spinner("🎤 Transcribing audio with Whisper..."):
                        from Scripts.speech.whisper_transcriber import WhisperTranscriber
                        whisper_model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
                        transcriber = WhisperTranscriber(model_size=whisper_model_size)
                        result = transcriber.transcribe(audio_bytes)  # Pass bytes directly
                        user_question = result["text"] if isinstance(result, dict) else str(result)
                    
                    # Show transcribed text
                    st.success("✅ Transcription Complete!")
                    st.write("**Transcribed Question:**")
                    st.info(user_question)
                    
            except ImportError:
                st.error("❌ Audio recorder not installed")
                st.info("Install it with: `pip install audio-recorder-streamlit`")
                st.info("Or use 'Upload Audio File' option instead")
            except FileNotFoundError as e:
                error_msg = str(e).lower()
                if 'ffmpeg' in error_msg:
                    st.error("❌ FFmpeg is not installed!")
                    st.warning("Whisper requires FFmpeg to process audio files.")
                    st.info("**Install FFmpeg:** See `INSTALL_FFMPEG.md` for instructions")
                    st.markdown("**Quick install:** `choco install ffmpeg` (requires Chocolatey)")
                else:
                    st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"Error recording/transcribing audio: {e}")
                error_msg = str(e).lower()
                if 'ffmpeg' in error_msg or 'winerror 2' in error_msg:
                    st.info("💡 This might be an FFmpeg issue. See `INSTALL_FFMPEG.md` for installation instructions.")
                else:
                    st.info("Make sure Whisper is installed: pip install openai-whisper")
                user_question = ""
        else:  # Upload Audio File
            st.info("📁 Upload an audio file with your question (MP3, WAV, M4A, etc.) - Supports English, Urdu, and Roman Urdu")
            audio_file = st.file_uploader(
                "Upload Audio File:",
                type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm'],
                help="Record or upload audio question in English, Urdu, or Roman Urdu"
            )
            
            if audio_file:
                try:
                    # Save uploaded file temporarily
                    import tempfile
                    import os
                    
                    # Create temp file and ensure it's properly saved
                    file_ext = os.path.splitext(audio_file.name)[1] or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, mode='wb') as tmp_file:
                        tmp_file.write(audio_file.read())
                        tmp_file.flush()
                        tmp_path = tmp_file.name
                    
                    # Ensure file exists and is readable
                    if not os.path.exists(tmp_path):
                        raise FileNotFoundError(f"Temporary file not created: {tmp_path}")
                    
                    try:
                        # Transcribe with Whisper
                        with st.spinner("🎤 Transcribing audio with Whisper..."):
                            from Scripts.speech.whisper_transcriber import WhisperTranscriber
                            whisper_model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
                            transcriber = WhisperTranscriber(model_size=whisper_model_size)
                            result = transcriber.transcribe(tmp_path)
                            user_question = result["text"] if isinstance(result, dict) else str(result)
                        
                        # Show transcribed text
                        st.success("✅ Transcription Complete!")
                        st.write("**Transcribed Question:**")
                        st.info(user_question)
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                    
                except FileNotFoundError as e:
                    error_msg = str(e).lower()
                    if 'ffmpeg' in error_msg:
                        st.error("❌ FFmpeg is not installed!")
                        st.warning("Whisper requires FFmpeg to process audio files.")
                        st.info("**Install FFmpeg:** See `INSTALL_FFMPEG.md` for instructions")
                        st.markdown("**Quick install:** `choco install ffmpeg` (requires Chocolatey)")
                    else:
                        st.error(f"Error: {e}")
                    user_question = ""
                except Exception as e:
                    st.error(f"Error transcribing audio: {e}")
                    error_msg = str(e).lower()
                    if 'ffmpeg' in error_msg or 'winerror 2' in error_msg:
                        st.info("💡 This might be an FFmpeg issue. See `INSTALL_FFMPEG.md` for installation instructions.")
                    else:
                        st.info("Make sure Whisper is installed: pip install openai-whisper")
                    user_question = ""
        
        if st.button("Query", type="primary"):
            if user_question.strip():
                try:
                    # Load RAG system
                    with st.spinner("Loading RAG system..."):
                        try:
                            from Scripts.rag.query_rag import RAGSystem
                            rag_system = RAGSystem(use_rag=True)
                        except ImportError as e:
                            st.error(f"RAG system import error: {e}")
                            st.info("Install missing packages: pip install langchain-core")
                            raise
                    
                    # Query
                    with st.spinner("Searching documents and generating answer..."):
                        result = rag_system.query(user_question)
                    
                    # Show results
                    st.success("✅ Query Complete!")
                    
                    st.subheader("💬 Answer")
                    st.info(result['answer'])
                    
                    # Show sources
                    if result.get('sources'):
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"Source {i}: {source.get('source', 'Unknown')}"):
                                st.write(f"**Document:** {source.get('source', 'Unknown')}")
                                st.write(f"**Page:** {source.get('page', 'N/A')}")
                                st.write(f"**Content:** {source.get('content', '')}")
                    else:
                        st.info("No sources retrieved")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Make sure RAG vectorstore is set up: python RunScripts/SETUP_RAG_WITH_DOCUMENTS.py")
            else:
                st.warning("⚠️ Please enter a question")
        
        # Example questions
        st.markdown("---")
        st.subheader("💡 Example Questions")
        example_questions = [
            "What is the emergency helpline number?",
            "What should I do during a flood?",
            "How to prepare for an earthquake?",
            "What emergency supplies should I have?",
            "Where are flood shelters located?",
            "What are the contact numbers for disaster management?",
        ]
        
        for q in example_questions:
            if st.button(f"📌 {q}", key=f"example_{q}"):
                st.session_state.rag_question = q
                st.rerun()

# ============================================================
# TASK 6: ABOUT
# ============================================================
elif task == "About":
    st.header("ℹ️ About RahatAI")
    
    st.markdown("""
    ### Multilingual Crisis Response NLP System
    
    RahatAI is a comprehensive NLP system for crisis and disaster management.
    
    **Components:**
    - **5 Classification Models**: Transformer (best), SVM, Naive Bayes, LSTM, CNN
    - **NER**: Extract locations, phone numbers, resources, persons, organizations
    - **Summarization**: Generate summaries of crisis reports using BART
    - **Misinformation Detection**: Identify false or misleading information
    - **RAG Pipeline**: Question answering with document retrieval
    - **🎤 Voice Input**: Whisper speech-to-text (available in Classification and RAG Query)
    
    **Best Model:** Transformer (73.35% accuracy, 0.7205 F1-score)
    **Production Model:** SVM (66.53% accuracy, fastest inference)
    
    **Dataset:** CrisisNLP + Kaggle (7,460 training samples, 6 categories)
    
    **Categories:**
    1. Affected individuals
    2. Donations and volunteering
    3. Infrastructure and utilities
    4. Not related or irrelevant
    5. Other Useful Information
    6. Sympathy and support
    
    **Languages Supported:** English, Urdu, Roman-Urdu
    
    ---
    
    **GitHub:** [github.com/malikkabdullah/RAHATAI](https://github.com/malikkabdullah/RAHATAI)
    
    **Documentation:** See `Docs/` folder for detailed guides
    """)
    
    # Model comparison
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Transformer 🏆", "73.35%", "Best Accuracy")
    with col2:
        st.metric("SVM ⭐", "66.53%", "Fast Production")
    with col3:
        st.metric("CNN", "52.07%", "GPU-Accelerated")
    with col4:
        st.metric("Naive Bayes", "48.76%", "Baseline")

# Footer
st.markdown("---")
st.caption("🆘 RahatAI - Multilingual Crisis Response NLP System | Developed for disaster management and emergency response")
