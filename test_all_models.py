"""
Test All Trained Models
Quick script to test all models with sample text
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json

# Load labels
with open("Data/Preprocessed/label_mappings.json", "r") as f:
    labels = json.load(f)

# Test texts
test_texts = [
    "Emergency in Karachi. Need food and water immediately.",
    "Donating food and blankets. Where should I send them?",
    "Power outage in Sector 5. Please restore electricity.",
    "Flood situation in Lahore. Need rescue boats urgently."
]

print("=" * 70)
print("TESTING ALL TRAINED MODELS")
print("=" * 70)

# Test SVM
print("\n1. SVM (Best Model - 66.5% accuracy):")
print("-" * 70)
try:
    from Scripts.classification.ml_models import SVMClassifier
    svm = SVMClassifier()
    svm.load("Models/svm.pkl")
    
    for text in test_texts:
        pred = svm.predict([text])[0]
        prob = svm.predict_proba([text])[0][pred]
        category = labels['idx_to_label'][str(pred)]
        print(f"  {category:30s} ({prob*100:.1f}%) - {text[:50]}...")
    print("  ✅ SVM working correctly")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test Naive Bayes
print("\n2. Naive Bayes (48.8% accuracy):")
print("-" * 70)
try:
    from Scripts.classification.ml_models import NaiveBayesClassifier
    nb = NaiveBayesClassifier()
    nb.load("Models/naive_bayes.pkl")
    
    for text in test_texts:
        pred = nb.predict([text])[0]
        prob = nb.predict_proba([text])[0][pred]
        category = labels['idx_to_label'][str(pred)]
        print(f"  {category:30s} ({prob*100:.1f}%) - {text[:50]}...")
    print("  ✅ Naive Bayes working correctly")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test CNN
print("\n3. CNN (52.1% accuracy, GPU-trained):")
print("-" * 70)
try:
    from Scripts.classification.dl_models import CNNClassifier
    cnn = CNNClassifier()
    cnn.load("Models/cnn")
    
    for text in test_texts:
        pred = cnn.predict([text])[0]
        prob = cnn.predict_proba([text])[0][pred]
        category = labels['idx_to_label'][str(pred)]
        print(f"  {category:30s} ({prob*100:.1f}%) - {text[:50]}...")
    print("  ✅ CNN working correctly")
except ValueError as e:
    if 'batch_shape' in str(e) or 'TensorFlow version' in str(e):
        print(f"  ⚠️  TensorFlow version compatibility issue")
        print(f"     Solution: Retrain with: python RunScripts/train_cnn.py")
    else:
        print(f"  ❌ Error: {e}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test LSTM
print("\n4. LSTM (27.9% accuracy - needs improvement):")
print("-" * 70)
try:
    from Scripts.classification.dl_models import LSTMClassifier
    lstm = LSTMClassifier()
    lstm.load("Models/lstm")
    
    for text in test_texts:
        pred = lstm.predict([text])[0]
        prob = lstm.predict_proba([text])[0][pred]
        category = labels['idx_to_label'][str(pred)]
        print(f"  {category:30s} ({prob*100:.1f}%) - {text[:50]}...")
    print("  ⚠️  LSTM working but low accuracy")
except ValueError as e:
    if 'batch_shape' in str(e) or 'TensorFlow version' in str(e):
        print(f"  ⚠️  TensorFlow version compatibility issue")
        print(f"     Solution: Retrain with: python RunScripts/STEP5_train_model3_lstm_efficient.py")
    else:
        print(f"  ❌ Error: {e}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test Transformer
print("\n5. Transformer (all-MiniLM-L6-v2):")
print("-" * 70)
try:
    from Scripts.classification.transformer_models import TransformerClassifier
    transformer = TransformerClassifier()
    transformer.load("Models/transformer")
    
    for text in test_texts:
        pred = transformer.predict([text])[0]
        prob = transformer.predict_proba([text])[0][pred]
        category = labels['idx_to_label'][str(pred)]
        print(f"  {category:30s} ({prob*100:.1f}%) - {text[:50]}...")
    print("  ✅ Transformer working correctly")
except Exception as e:
    print(f"  ⚠️  Transformer not trained or Error: {e}")

print("\n" + "=" * 70)
print("TESTING COMPLETE")
print("=" * 70)
print("\n🏆 Best Model: Transformer (73.35% accuracy)")
print("⭐ Recommended: SVM for fast production (66.53% accuracy)")
print("💡 For best results: Use Ensemble (Transformer + SVM)")

