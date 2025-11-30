"""
Transformer-based Models for Classification

Implements 1 Transformer model (required):
- all-MiniLM-L6-v2: Lightweight sentence transformer for text classification
- Supports English, Urdu, Roman-Urdu text
- Uses sentence-transformers for embeddings + simple classifier
- Much faster and simpler than XLM-RoBERTa
"""
import numpy as np
import os

# Disable TensorFlow to prevent conflicts with DirectML
# We're using sentence-transformers (PyTorch-based), so TF is not needed
# Set these BEFORE any imports that might trigger TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DISABLE_SEGMENT_REDUCTION_OP", "1")

# Suppress TensorFlow warnings/errors about DirectML registration
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*DirectML.*')
warnings.filterwarnings('ignore', message='.*platform is already registered.*')

# Import sentence-transformers (PyTorch-based, no TensorFlow needed)
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pickle
from typing import List, Dict, Optional
from pathlib import Path

# Removed TextClassificationDataset - not needed for sentence-transformers approach

class TransformerClassifier:
    """
    Transformer-based Text Classifier using all-MiniLM-L6-v2
    Uses sentence-transformers for embeddings + simple classifier on top
    Much faster and simpler than XLM-RoBERTa
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 num_labels: int = None,
                 max_length: int = 128,
                 device: str = None,
                 classifier_type: str = "mlp"):  # "mlp" or "logistic"
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.classifier_type = classifier_type
        
        # Initialize sentence transformer model (much simpler!)
        print(f"Loading sentence transformer: {model_name}")
        print("  (This is much smaller and faster than XLM-RoBERTa)")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Classifier will be trained on embeddings
        self.classifier = None
        self.is_trained = False
    
    def _get_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Get embeddings for texts using sentence transformer"""
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def train(self, X_train: List[str], y_train: np.ndarray,
              X_val: Optional[List[str]] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 10,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              output_dir: str = "Models/transformer",
              save_steps: int = 500) -> Dict:
        """
        Train the transformer classifier
        Uses sentence-transformers for embeddings + simple classifier
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional, for evaluation)
            y_val: Validation labels (optional)
            epochs: Number of epochs (for MLP classifier)
            batch_size: Batch size for embedding generation
            learning_rate: Learning rate (for MLP classifier)
            output_dir: Directory to save model
            save_steps: Not used (kept for compatibility)
            
        Returns:
            dict: Training history (simplified)
        """
        print("Training Transformer Classifier with all-MiniLM-L6-v2...")
        print("  Step 1: Generating embeddings for training data...")
        
        num_labels = len(np.unique(y_train))
        self.num_labels = num_labels
        
        # Step 1: Generate embeddings for training data
        train_embeddings = self._get_embeddings(X_train, batch_size=batch_size, show_progress=True)
        print(f"  Generated {len(train_embeddings)} embeddings of dimension {train_embeddings.shape[1]}")
        
        # Step 2: Train classifier on embeddings
        print("  Step 2: Training classifier on embeddings...")
        
        if self.classifier_type == "mlp":
            # Use MLP classifier (neural network)
            # Reduced hidden layer sizes to prevent OOM
            # When validation data is provided, disable early_stopping's internal validation
            # We'll evaluate on our own validation set instead
            mlp_params = {
                'hidden_layer_sizes': (128, 64),  # Reduced from (256, 128) to save memory
                'max_iter': epochs,
                'learning_rate_init': learning_rate,
                'verbose': True,
                'random_state': 42,
                'batch_size': min(200, len(train_embeddings))  # Limit batch size for MLP training
            }
            
            # Only add early_stopping and validation_fraction if no validation data provided
            if X_val is None:
                mlp_params['early_stopping'] = True
                mlp_params['validation_fraction'] = 0.1
            else:
                mlp_params['early_stopping'] = False
                # Don't set validation_fraction - let it use default (0.0)
            
            self.classifier = MLPClassifier(**mlp_params)
        else:
            # Use Logistic Regression (faster, simpler)
            self.classifier = LogisticRegression(
                max_iter=epochs * 100,  # Logistic regression needs more iterations
                C=1.0,
                random_state=42,
                verbose=1 if epochs > 1 else 0
            )
        
        # Train classifier
        if X_val is not None and y_val is not None:
            # Generate validation embeddings
            print("  Step 3: Generating embeddings for validation data...")
            val_embeddings = self._get_embeddings(X_val, batch_size=batch_size, show_progress=True)
            
            # Train with validation
            self.classifier.fit(train_embeddings, y_train)
            
            # Evaluate on validation
            val_score = self.classifier.score(val_embeddings, y_val)
            print(f"  Validation accuracy: {val_score:.4f}")
        else:
            # Train without validation
            self.classifier.fit(train_embeddings, y_train)
        
        self.is_trained = True
        print("  ✓ Transformer training completed!")
        
        # Return simplified history
        history = {
            'loss': [0.0],  # Placeholder
            'train_loss': [0.0],
            'val_loss': [0.0] if X_val is not None else None
        }
        
        return history
    
    def predict(self, X: List[str], batch_size: int = 32) -> np.ndarray:
        """Predict labels"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Generate embeddings
        embeddings = self._get_embeddings(X, batch_size=batch_size, show_progress=False)
        
        # Predict using classifier
        predictions = self.classifier.predict(embeddings)
        
        return predictions
    
    def predict_proba(self, X: List[str], batch_size: int = 32) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Generate embeddings
        embeddings = self._get_embeddings(X, batch_size=batch_size, show_progress=False)
        
        # Get probabilities from classifier
        probabilities = self.classifier.predict_proba(embeddings)
        
        return probabilities
    
    def save(self, path: str):
        """Save embedding model and classifier"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.classifier is not None:
            # Save sentence transformer (it will save itself)
            embedding_path = path / "embedding_model"
            self.embedding_model.save(str(embedding_path))
            
            # Save classifier
            classifier_path = path / "classifier.pkl"
            with open(classifier_path, 'wb') as f:
                pickle.dump({
                    'classifier': self.classifier,
                    'classifier_type': self.classifier_type,
                    'num_labels': self.num_labels,
                    'model_name': self.model_name
                }, f)
            
            print(f"Model saved to {path}")
        else:
            raise ValueError("Model not trained yet")
    
    def load(self, path: str):
        """Load embedding model and classifier"""
        path = Path(path)
        
        # Load sentence transformer
        embedding_path = path / "embedding_model"
        if embedding_path.exists():
            self.embedding_model = SentenceTransformer(str(embedding_path))
        else:
            # Fallback to original model name
            self.embedding_model = SentenceTransformer(self.model_name)
        
        # Load classifier
        classifier_path = path / "classifier.pkl"
        with open(classifier_path, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.classifier_type = data.get('classifier_type', 'mlp')
            self.num_labels = data.get('num_labels', None)
        
        self.is_trained = True
        print(f"Model loaded from {path}")

