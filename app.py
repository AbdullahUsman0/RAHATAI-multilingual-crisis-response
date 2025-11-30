"""
RahatAI - Simple Demo App
Multilingual Crisis Response NLP System
"""

import streamlit as st
import sys
from pathlib import Path
import json
import pickle

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="RahatAI - Crisis Response NLP",
    page_icon="🆘",
    layout="wide"
)

# Title
st.title("🆘 RahatAI - Crisis Response NLP System")
st.markdown("**Multilingual Crisis Response & Disaster Management**")
st.markdown("---")

# Sidebar - Model Selection
st.sidebar.header("⚙ Settings")
task = st.sidebar.selectbox(
    "Choose Task:",
    ["Classification", "NER", "Summarization", "Misinformation Detection", "About"]
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
    st.write("Classify crisis-related text into 6 categories using the best model (SVM - 66.5% accuracy)")
    
    # Input
    user_text = st.text_area(
        "Enter crisis text:",
        placeholder="Example: Emergency in Karachi. Need food and water. Contact 0300-1234567.",
        height=100
    )
    
    if st.button("Classify"):
        if user_text.strip():
            try:
                # Load SVM model
                with st.spinner("Loading model..."):
                    from Scripts.classification.ml_models import SVMClassifier
                    svm = SVMClassifier()
                    svm.load("Models/svm.pkl")
                
                # Predict
                with st.spinner("Classifying..."):
                    prediction = svm.predict([user_text])[0]
                    probabilities = svm.predict_proba([user_text])[0]
                
                # Show results
                st.success("Classification Complete!")
                st.write(f"**Category:** {idx_to_label[str(prediction)]}")
                
                # Show probabilities
                st.write("**Confidence Scores:**")
                prob_data = {idx_to_label[str(i)]: f"{prob*100:.1f}%" for i, prob in enumerate(probabilities)}
                st.json(prob_data)
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure the SVM model is trained: python RunScripts/STEP4_train_model2_svm.py")
        else:
            st.warning("Please enter some text")

# ============================================================
# TASK 2: NER (Named Entity Recognition)
# ============================================================
elif task == "NER":
    st.header("🏷 Named Entity Recognition")
    st.write("Extract locations, phone numbers, resources, persons, and organizations from text")
    
    # Input
    user_text = st.text_area(
        "Enter crisis text:",
        placeholder="Example: Emergency in Lahore. Contact Dr. Ahmed at 0300-1234567. Need food and medical supplies.",
        height=100
    )
    
    if st.button("Extract Entities"):
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
                st.success("Extraction Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📍 Locations:**")
                    st.write(", ".join(entities['locations']) if entities['locations'] else "None found")
                    
                    st.write("**📞 Phone Numbers:**")
                    st.write(", ".join(entities['phone_numbers']) if entities['phone_numbers'] else "None found")
                    
                    st.write("**📦 Resources:**")
                    st.write(", ".join(entities['resources']) if entities['resources'] else "None found")
                
                with col2:
                    st.write("**👤 Persons:**")
                    st.write(", ".join(entities['persons']) if entities['persons'] else "None found")
                    
                    st.write("**🏢 Organizations:**")
                    st.write(", ".join(entities['organizations']) if entities['organizations'] else "None found")
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text")

# ============================================================
# TASK 3: SUMMARIZATION
# ============================================================
elif task == "Summarization":
    st.header("📝 Text Summarization")
    st.write("Generate concise summaries of crisis reports")
    
    # Input
    user_text = st.text_area(
        "Enter text to summarize:",
        placeholder="Paste long crisis report here...",
        height=200
    )
    
    max_length = st.slider("Summary length:", 50, 300, 150)
    
    if st.button("Summarize"):
        if user_text.strip():
            if len(user_text) < 50:
                st.warning("Text too short to summarize. Add more content.")
            else:
                try:
                    # Load summarizer
                    with st.spinner("Loading summarization model..."):
                        from Scripts.summarization.summarizer import CrisisSummarizer
                        summarizer = CrisisSummarizer()
                    
                    # Summarize
                    with st.spinner("Generating summary..."):
                        summary = summarizer.summarize(user_text, max_length=max_length, min_length=30)
                    
                    # Show results
                    st.success("Summarization Complete!")
                    st.write("**Summary:**")
                    st.info(summary)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text")

# ============================================================
# TASK 4: MISINFORMATION DETECTION
# ============================================================
elif task == "Misinformation Detection":
    st.header("🔍 Misinformation Detection")
    st.write("Detect if crisis information is verified or potentially false")
    
    # Input
    user_text = st.text_area(
        "Enter crisis text to verify:",
        placeholder="Example: Rumor says cyclone approaching. Unconfirmed reports.",
        height=100
    )
    
    if st.button("Detect"):
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
                st.success("Detection Complete!")
                
                if prediction == 1:
                    st.error(f"⚠ Potential Misinformation (Confidence: {probabilities[1]*100:.1f}%)")
                else:
                    st.success(f"✓ Likely Verified (Confidence: {probabilities[0]*100:.1f}%)")
                
                # Show linguistic features
                features = detector.extract_linguistic_features(user_text)
                
                with st.expander("View Analysis Details"):
                    st.write("**Linguistic Features:**")
                    st.write(f"- Uncertainty markers: {features['uncertainty_count']}")
                    st.write(f"- Credibility markers: {features['credibility_count']}")
                    st.write(f"- Exclamation marks: {features['exclamation_count']}")
                    st.write(f"- Question marks: {features['question_count']}")
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text")

# ============================================================
# TASK 5: ABOUT
# ============================================================
elif task == "About":
    st.header("ℹ About RahatAI")
    
    st.markdown("""
    ### Multilingual Crisis Response NLP System
    
    RahatAI is a comprehensive NLP system for crisis and disaster management.
    
    **Components:**
    - **5 Classification Models**: Transformer (best), SVM, Naive Bayes, LSTM, CNN
    - **NER**: Extract locations, phone numbers, resources, persons, organizations
    - **Summarization**: Generate summaries of crisis reports
    - **Misinformation Detection**: Identify false or misleading information
    - **RAG Pipeline**: Question answering with document retrieval
    
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
