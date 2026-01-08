 🆘 RahatAI - Multilingual Crisis Response NLP System

<div align="center">

![RahatAI](https://img.shields.io/badge/RahatAI-Crisis%20Response-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A comprehensive NLP system for crisis and disaster management with multilingual support**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

RahatAI is an advanced Natural Language Processing system designed specifically for crisis response and disaster management. It provides real-time analysis of crisis-related text in multiple languages (English, Urdu, Roman-Urdu) with state-of-the-art machine learning models.

### Key Capabilities

- **Text Classification**: Categorize crisis messages into 6 predefined categories
- **Named Entity Recognition**: Extract locations, phone numbers, resources, persons, and organizations
- **Text Summarization**: Generate concise summaries of crisis reports using BART
- **Misinformation Detection**: Identify potentially false or misleading information
- **RAG Query System**: Answer questions using official disaster response documents
- **Voice Input**: Speech-to-text transcription using OpenAI Whisper

---

## ✨ Features

### 🎯 Classification Models
- **Transformer** (Best Performance): 73.35% accuracy, 0.7205 F1-score
- **SVM** (Production): 66.53% accuracy, fastest inference
- **Naive Bayes** (Baseline): 48.76% accuracy
- **LSTM** (Deep Learning): ~60% accuracy
- **CNN** (Deep Learning): ~52% accuracy

### 🏷️ Named Entity Recognition
Extract critical information from crisis text:
- 📍 Locations
- 📞 Phone Numbers
- 📦 Resources
- 👤 Persons
- 🏢 Organizations

### 📝 Summarization
- Uses Facebook BART model for abstractive summarization
- Configurable min/max length
- Handles long crisis reports efficiently

### 🔍 Misinformation Detection
- Identifies potentially false information
- Analyzes linguistic features (uncertainty markers, credibility indicators)
- Provides confidence scores

### 💬 RAG Query System
- Question answering from official disaster response documents
- Document retrieval with source attribution
- Supports voice queries

### 🎤 Voice Input
- Browser-based audio recording
- Audio file upload support
- Whisper speech-to-text transcription
- Supports English, Urdu, and Roman Urdu

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for Whisper audio processing)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/AbdullahUsman0/RAHATAI-multilingual-crisis-response.git
cd RAHATAI-multilingual-crisis-response
```

### Step 2: Install Dependencies

```bash
pip install -r extras/requirements.txt
```

### Step 3: Install FFmpeg (Required for Voice Input)

**Windows (using Chocolatey):**
```bash
choco install ffmpeg
```

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

Alternatively, set the `FFMPEG_PATH` environment variable if FFmpeg is installed in a custom location.

### Step 4: Configure Environment Variables (Optional)

```bash
# Set FFmpeg path (if not in system PATH)
export FFMPEG_PATH="/path/to/ffmpeg/bin"

# Set Whisper model size (default: "base")
export WHISPER_MODEL_SIZE="base"  # Options: tiny, base, small, medium, large
```

---

## 💻 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Training Models

Train individual models using the scripts in `RunScripts/`:

```bash
# Train SVM (fastest)
python RunScripts/train_svm.py

# Train Transformer (best accuracy)
python RunScripts/train_transformer.py

# Train Naive Bayes
python RunScripts/train_naive_bayes.py

# Train LSTM
python RunScripts/train_lstm_efficient.py

# Train CNN
python RunScripts/train_cnn.py
```

### Setting Up RAG System

```bash
python RunScripts/SETUP_RAG_WITH_DOCUMENTS.py
```

---

## 📊 Models

### Classification Categories

1. **Affected individuals** - People in need of immediate assistance
2. **Donations and volunteering** - Offers of help and resources
3. **Infrastructure and utilities** - Damage to buildings, roads, utilities
4. **Not related or irrelevant** - Non-crisis related content
5. **Other Useful Information** - General crisis-related information
6. **Sympathy and support** - Emotional support messages

### Model Performance

| Model | Accuracy | F1-Score | Use Case |
|-------|----------|----------|----------|
| Transformer 🏆 | 73.35% | 0.7205 | Best overall performance |
| SVM ⭐ | 66.53% | - | Production (fastest) |
| LSTM | ~60% | - | Deep learning option |
| CNN | 52.07% | - | GPU-accelerated |
| Naive Bayes | 48.76% | - | Baseline |

### Dataset

- **Training Samples**: 7,460
- **Sources**: CrisisNLP + Kaggle
- **Languages**: English, Urdu, Roman-Urdu
- **Categories**: 6

---

## 🏗️ Project Structure

```
RAHATAI-multilingual-crisis-response/
├── app.py                      # Main Streamlit application
├── Scripts/                    # Core functionality modules
│   ├── classification/         # Classification models
│   ├── ner/                    # Named Entity Recognition
│   ├── summarization/          # Text summarization
│   ├── misinformation/        # Misinformation detection
│   ├── rag/                    # RAG query system
│   ├── speech/                 # Speech-to-text (Whisper)
│   └── utils/                  # Utility functions
├── RunScripts/                 # Training and setup scripts
├── Models/                     # Trained model files (not in repo)
├── Data/                       # Datasets (not in repo)
│   ├── Preprocessed/           # Preprocessed data
│   └── documents/              # RAG document sources
├── Outputs/                    # Model outputs and plots
├── extras/                     # Additional resources
│   ├── requirements.txt        # Python dependencies
│   └── api_server.py           # REST API server
└── README.md                   # This file
```

---

## 🔧 Configuration

### Environment Variables

- `FFMPEG_PATH`: Path to FFmpeg binary directory (optional)
- `WHISPER_MODEL_SIZE`: Whisper model size - `tiny`, `base`, `small`, `medium`, `large` (default: `base`)

### Model Files

Model files are not included in the repository due to size. You need to train them using the scripts in `RunScripts/` or download pre-trained models separately.

---

## 📚 Documentation

Detailed documentation is available in the `Docs/` folder (if available) and inline code comments.

### Key Components

- **Classification**: Multi-model text classification system
- **NER**: Multilingual named entity recognition
- **Summarization**: BART-based abstractive summarization
- **Misinformation Detection**: Linguistic feature-based detection
- **RAG**: Retrieval-Augmented Generation for document Q&A

---

## 🌐 Languages Supported

- ✅ English
- ✅ Urdu
- ✅ Roman-Urdu

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Abdullah Usman**

- GitHub: [@AbdullahUsman0](https://github.com/AbdullahUsman0)
- Repository: [RAHATAI-multilingual-crisis-response](https://github.com/AbdullahUsman0/RAHATAI-multilingual-crisis-response)

---

## 🙏 Acknowledgments

- CrisisNLP dataset providers
- Kaggle community
- OpenAI Whisper team
- Hugging Face Transformers
- Streamlit team

---

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

<div align="center">

**🆘 Built for disaster management and emergency response**

Made with ❤️ for crisis response teams worldwide

</div>

