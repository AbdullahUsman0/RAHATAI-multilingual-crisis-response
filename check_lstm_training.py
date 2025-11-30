"""
Script to verify if LSTM training completed successfully
"""
from pathlib import Path
from datetime import datetime
import json

print("=" * 70)
print("LSTM TRAINING VERIFICATION")
print("=" * 70)

# Check model files
lstm_model = Path("Models/lstm/model.h5")
lstm_tokenizer = Path("Models/lstm/tokenizer.pkl")
lstm_metrics = Path("Outputs/results/lstm_test_metrics.json")

print("\n1. MODEL FILES:")
print("-" * 70)

if lstm_model.exists():
    stat = lstm_model.stat()
    size_mb = stat.st_size / 1024 / 1024
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    time_ago = datetime.now() - mod_time
    
    print(f"   ✓ model.h5 exists")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Modified {time_ago.days} days, {time_ago.seconds // 3600} hours ago")
else:
    print("   ✗ model.h5 NOT FOUND")

if lstm_tokenizer.exists():
    stat = lstm_tokenizer.stat()
    size_kb = stat.st_size / 1024
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    print(f"   ✓ tokenizer.pkl exists ({size_kb:.2f} KB)")
else:
    print("   ✗ tokenizer.pkl NOT FOUND")

# Check metrics
print("\n2. TEST METRICS:")
print("-" * 70)

if lstm_metrics.exists():
    with open(lstm_metrics, 'r') as f:
        metrics = json.load(f)
    
    print(f"   ✓ Test metrics file exists")
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   F1 Score: {metrics['f1']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    
    # Check confusion matrix
    cm = metrics['confusion_matrix']
    print(f"\n   Confusion Matrix Analysis:")
    
    # Count how many classes are being predicted
    predicted_classes = set()
    for row in cm:
        for i, val in enumerate(row):
            if val > 0:
                predicted_classes.add(i)
    
    print(f"   Classes being predicted: {sorted(predicted_classes)}")
    
    # Check if model is only predicting one class
    if len(predicted_classes) == 1:
        print(f"   ⚠️  WARNING: Model is only predicting class {list(predicted_classes)[0]}")
        print(f"      This suggests training may not have completed properly")
    else:
        print(f"   ✓ Model predicts {len(predicted_classes)} different classes")
    
    # Check if most predictions are for one class
    total_predictions = sum(sum(row) for row in cm)
    if total_predictions > 0:
        max_class = max(range(len(cm[0])), key=lambda i: sum(row[i] for row in cm))
        max_predictions = sum(row[max_class] for row in cm)
        max_percentage = (max_predictions / total_predictions) * 100
        
        if max_percentage > 80:
            print(f"   ⚠️  WARNING: {max_percentage:.1f}% of predictions are for class {max_class}")
            print(f"      Model may be overfitting to one class")
    
else:
    print("   ✗ Test metrics file NOT FOUND")
    print("   This means training did NOT complete successfully")

# Check plots
print("\n3. VISUALIZATION PLOTS:")
print("-" * 70)

plots_dir = Path("Outputs/plots")
required_plots = [
    "lstm_confusion_matrix.png",
    "lstm_accuracy.png",
    "lstm_loss.png"
]

for plot in required_plots:
    plot_path = plots_dir / plot
    if plot_path.exists():
        print(f"   ✓ {plot} exists")
    else:
        print(f"   ✗ {plot} NOT FOUND")

# Summary
print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)

if lstm_model.exists() and lstm_tokenizer.exists() and lstm_metrics.exists():
    print("\n✓ LSTM model files exist")
    print("✓ Test metrics exist")
    
    if len(predicted_classes) == 1:
        print("\n⚠️  WARNING: Training appears incomplete or model didn't learn properly")
        print("   The model is only predicting one class")
        print("   Recommendation: Retrain the LSTM model")
        print("\n   Run: python RunScripts/STEP5_train_model3_lstm_efficient.py")
    elif metrics['accuracy'] < 0.30:
        print("\n⚠️  WARNING: Model performance is very low")
        print("   Accuracy is below 30%")
        print("   Recommendation: Consider retraining with different parameters")
    else:
        print("\n✓ Training appears to have completed")
        print("  (Performance may need improvement)")
else:
    print("\n✗ Training did NOT complete successfully")
    print("  Missing required files")
    print("\n  Run: python RunScripts/STEP5_train_model3_lstm_efficient.py")

print("=" * 70)


