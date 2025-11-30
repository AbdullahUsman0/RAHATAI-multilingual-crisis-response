"""
STEP 7: TRAIN MODEL 5 - TRANSFORMER (all-MiniLM-L6-v2)
Lightweight sentence transformer model for text classification
Much faster and simpler than XLM-RoBERTa
"""
# IMPORTANT: Set environment variables FIRST, before ANY imports
# This prevents TensorFlow from initializing and conflicting with DirectML
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TF logs
os.environ["TRANSFORMERS_NO_TF"] = "1"   # Disable TF backend  
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN
# Prevent TensorFlow from loading/registering DirectML (causes conflicts)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Don't use CUDA
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

# ----------------------------------------------------
# Import all required libraries
# ----------------------------------------------------
from pathlib import Path
import sys

# Setting project root path so Python can find modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Basic imports
import numpy as np
import pandas as pd

# Importing custom-built scripts
# Note: TensorFlow DirectML registration errors are harmless - we use PyTorch
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow fatal errors by redirecting stderr temporarily
import sys
from contextlib import redirect_stderr
from io import StringIO

# Import with error suppression (TensorFlow errors are harmless for us)
try:
    with redirect_stderr(StringIO()):
        from Scripts.classification.transformer_models import TransformerClassifier
        from Scripts.evaluation.metrics import (
            calculate_all_metrics,
            plot_confusion_matrix,
            plot_training_history,
            save_metrics,
            print_metrics_summary,
        )
except Exception as e:
    # If import fails due to TensorFlow error, try again without suppression
    # The error is usually just a warning about DirectML registration
    print("Note: TensorFlow DirectML warning (harmless - we use PyTorch)")
    from Scripts.classification.transformer_models import TransformerClassifier
    from Scripts.evaluation.metrics import (
        calculate_all_metrics,
        plot_confusion_matrix,
        plot_training_history,
        save_metrics,
        print_metrics_summary,
    )

# Configure DirectML for AMD GPU support
# torch-directml will be used automatically if available

print("=" * 70)
print("STEP 7: TRAINING MODEL 5 - TRANSFORMER (all-MiniLM-L6-v2)")
print("Lightweight sentence transformer - Fast and efficient")
print("=" * 70)

# ----------------------------------------------------
# 1. Load preprocessed data
# ----------------------------------------------------
print("\n1. Loading preprocessed data...")

# Load CSV files
train_df = pd.read_csv("Data/Preprocessed/train_preprocessed.csv")
val_df = pd.read_csv("Data/Preprocessed/val_preprocessed.csv")
test_df = pd.read_csv("Data/Preprocessed/test_preprocessed.csv")

# Print dataset sizes
print("   Training samples   :", len(train_df))
print("   Validation samples :", len(val_df))
print("   Test samples       :", len(test_df))

# Extract text and labels from dataframes
X_train = train_df["text"].tolist()
y_train = train_df["label_encoded"].values

X_val = val_df["text"].tolist()
y_val = val_df["label_encoded"].values

X_test = test_df["text"].tolist()
y_test = test_df["label_encoded"].values

# Count number of classes
unique_classes = np.unique(y_train)
num_classes = len(unique_classes)
print("   Number of classes  :", num_classes)

# ----------------------------------------------------
# 2. Create Transformer model
# ----------------------------------------------------
print("\n2. Creating Transformer model (all-MiniLM-L6-v2)...")
print("   Using sentence-transformers - much faster and simpler!")

# Create transformer model with all-MiniLM-L6-v2
# This is much smaller (~80MB vs ~1GB) and faster to train
transformer_model = TransformerClassifier(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    num_labels=num_classes,
    max_length=128,
    device=None,  # Not needed for sentence-transformers
    classifier_type="mlp"  # Use MLP classifier (or "logistic" for faster training)
)

# ----------------------------------------------------
# 3. Train model
# ----------------------------------------------------
print("\n3. Training Transformer model...")
print("   Note: First run will download the model (~80MB - much smaller!)")
print("   This is much faster than XLM-RoBERTa training")

# Train the model
# all-MiniLM-L6-v2 training is much faster - no complex transformer training needed
# Reduced batch size and epochs to prevent OOM errors
history = transformer_model.train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=8,  # Reduced from 50 to prevent OOM
    batch_size=16,  # Reduced from 32 to prevent OOM (smaller batches = less memory)
    learning_rate=0.001,  # Standard learning rate for MLP
    output_dir="Models/transformer"
)

# ----------------------------------------------------
# 4. Evaluate on test set
# ----------------------------------------------------
print("\n4. Evaluating on TEST set...")

# Predict class labels
preds = transformer_model.predict(X_test, batch_size=16)  # Reduced to prevent OOM

# Predict probabilities
probs = transformer_model.predict_proba(X_test, batch_size=16)  # Reduced to prevent OOM

# Compute all evaluation metrics
metrics = calculate_all_metrics(
    y_test,
    preds,
    probs,
    labels=list(range(num_classes))
)

# Print summary in simple format
print_metrics_summary(metrics)

# ----------------------------------------------------
# 5. Save model, metrics, and plots
# ----------------------------------------------------
print("\n5. Saving model and results...")

# Create folders if not exist
models_dir = Path("Models/transformer")
plots_dir = Path("Outputs/plots")
results_dir = Path("Outputs/results")

models_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Save model (already saved during training, but save again to be sure)
transformer_model.save(str(models_dir))

# Save test metrics
save_metrics(metrics, results_dir / "transformer_test_metrics.json")

# Save confusion matrix plot
plot_confusion_matrix(
    y_test,
    preds,
    labels=list(range(num_classes)),
    save_path=plots_dir / "transformer_confusion_matrix.png",
    title="Transformer (all-MiniLM-L6-v2) - Confusion Matrix (Test Set)"
)

# Save training curves if history available
if history:
    plot_training_history(
        history,
        str(plots_dir),
        "transformer"
    )

# ----------------------------------------------------
# Summary
# ----------------------------------------------------
print("\n" + "=" * 70)
print("MODEL 5 TRAINING COMPLETE - TRANSFORMER (all-MiniLM-L6-v2)")
print("=" * 70)



