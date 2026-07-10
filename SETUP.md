# AI Fake News Detection Platform - Setup Guide

## Overview

This is a production-ready full-stack AI application that classifies news articles as **Real** or **Fake** using Transformer-based NLP models with Explainable AI (XAI) capabilities.

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Python 3.12+** - Programming language
- **Transformers** - HuggingFace NLP models
- **PyTorch** - Deep learning framework
- **NLTK/SpaCy** - Text preprocessing

### Frontend
- **Next.js 16** - React framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS
- **Lucide React** - Icon library
- **Framer Motion** - Animation library

## Prerequisites

- Python 3.12 or higher
- Node.js 18 or higher
- npm or yarn
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "AI Fake News Prediction Model"
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Download NLTK Data

The first time you run the application, NLTK will automatically download required data. You can also download it manually:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

#### Optional: Download SpaCy Model

For enhanced text preprocessing:

```bash
python -m spacy download en_core_web_sm
```

### 3. Frontend Setup

#### Navigate to Frontend Directory

```bash
cd frontend
```

#### Install Node Dependencies

```bash
npm install
```

## Running the Application

### WELFake Dataset And Training

The classifier is trained on the WELFake dataset. WELFake encodes `1` as fake and `0` as real; the training pipeline maps these values to the API convention of `0 = fake` and `1 = real`. Download the public dataset and train the model before starting the API:

```powershell
.\download_welfake.ps1
.\.venv\Scripts\python.exe .\training\train.py
```

The training command removes exact duplicate articles before making an 80/20 stratified holdout split. It saves the trained model, vectorizer, and the resulting WELFake evaluation metrics in `models/`.

### Persistent Windows Launcher

To start both services in separate PowerShell windows and open the dashboard:

```powershell
.\run_project.ps1
```

This is the recommended local development command when launching through an IDE or automation tool that cleans up child processes. The backend and frontend remain active until their respective PowerShell windows are closed.

The live API uses the local WELFake-trained model; no Hugging Face token is required.

### Start Backend Server

Open a terminal in the project root (with virtual environment activated):

```bash
python backend/main.py
```

The backend will start on `http://127.0.0.1:8000`

API documentation will be available at `http://127.0.0.1:8000/docs`

### Start Frontend Development Server

Open a new terminal in the `frontend` directory:

```bash
npm run dev
```

The frontend will start on `http://localhost:3000`

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Paste a news article or headline into the text area
3. Click "Analyze Article" to get:
   - **Prediction**: Real or Fake classification
   - **Confidence Score**: Model's confidence level
   - **Explainable AI**: Word-level attribution highlighting
   - **Keywords**: Extracted key terms
   - **Topic**: Article topic classification
   - **Sentiment**: Emotional tone analysis
   - **Summary**: AI-generated summary

## API Endpoints

### POST /predict
Classify text as Real or Fake news

**Request:**
```json
{
  "text": "Your news article text here",
  "model_type": "transformer"
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 0.85,
  "model_used": "DeBERTa-v3"
}
```

### POST /explain
Get word-level attributions for prediction

**Request:**
```json
{
  "text": "Your news article text here",
  "model_type": "transformer"
}
```

**Response:**
```json
{
  "attributions": [
    {"token": "word1", "score": 0.5},
    {"token": "word2", "score": -0.3}
  ]
}
```

### POST /summary
Generate article summary

**Request:**
```json
{
  "text": "Your news article text here"
}
```

**Response:**
```json
{
  "summary": "Generated summary text"
}
```

### POST /keywords
Extract key keywords

**Request:**
```json
{
  "text": "Your news article text here"
}
```

**Response:**
```json
{
  "keywords": ["keyword1", "keyword2", "keyword3"]
}
```

### POST /topics
Classify article topic

**Request:**
```json
{
  "text": "Your news article text here"
}
```

**Response:**
```json
{
  "topic": "Politics"
}
```

### POST /sentiment
Analyze sentiment

**Request:**
```json
{
  "text": "Your news article text here"
}
```

**Response:**
```json
{
  "sentiment": "NEGATIVE",
  "pos": 0.1,
  "neg": 0.7,
  "neu": 0.2
}
```

### GET /metrics
Get model performance metrics

**Response:**
```json
{
  "accuracy": 0.92,
  "precision": 0.91,
  "recall": 0.90,
  "f1_score": 0.905
}
```

## Project Structure

```
AI Fake News Prediction Model/
├── backend/
│   └── main.py                 # FastAPI application
├── frontend/
│   ├── app/                    # Next.js app directory
│   │   ├── page.tsx           # Main page
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   ├── components/            # React components
│   │   ├── Dashboard.tsx      # Main dashboard
│   │   ├── ExplainableText.tsx # XAI component
│   │   └── ModelMetrics.tsx   # Metrics display
│   └── package.json           # Node dependencies
├── models/
│   └── model_loader.py        # Model loading and prediction
├── preprocessing/
│   └── text_preprocessor.py   # Text preprocessing
├── explainability/
│   └── explainer.py           # XAI implementation
├── datasets/                  # Dataset storage
├── training/                  # Training scripts
├── requirements.txt           # Python dependencies
└── SETUP.md                   # This file
```

## Model Information

### Default Model: DeBERTa-v3
The application uses Microsoft's DeBERTa-v3-base model by default, which is state-of-the-art for text classification tasks.

### Fallback Mode
If the Transformer model fails to load, the application automatically falls back to a heuristic-based classifier that uses keyword matching.

### Available Models
- DeBERTa-v3 (default, recommended)
- RoBERTa
- BERT
- DistilBERT (lighter, faster)

## Troubleshooting

### Backend Issues

**Model loading fails:**
- The application will automatically use fallback mode
- Check your internet connection (first run downloads model)
- Ensure you have enough disk space (~500MB for model)

**Port already in use:**
- Change the port in `backend/main.py`:
  ```python
  uvicorn.run(app, host="0.0.0.0", port=8001)  # Change 8000 to 8001
  ```
- Update the frontend API URL in `frontend/components/Dashboard.tsx` and `frontend/app/page.tsx`

### Frontend Issues

**Dependencies not installing:**
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`
- Run `npm install` again

**CORS errors:**
- Ensure backend is running before starting frontend
- Check that CORS is enabled in `backend/main.py`

## Development

### Adding New Features

1. **Backend**: Add new endpoints in `backend/main.py`
2. **Frontend**: Create components in `frontend/components/`
3. **Models**: Add new model architectures in `models/model_loader.py`
4. **Preprocessing**: Extend `preprocessing/text_preprocessor.py`

### Training Custom Models

Place your training scripts in the `training/` directory. The model loader supports loading custom trained models.

## Production Deployment

### Backend
- Use Gunicorn or Uvicorn with multiple workers
- Set up environment variables for configuration
- Use a production WSGI server

### Frontend
- Build the application: `npm run build`
- Deploy the `.next` folder to your hosting platform
- Set up environment variables for API URLs

## License

MIT License

## Support

For issues or questions, please refer to the project documentation or create an issue in the repository.
