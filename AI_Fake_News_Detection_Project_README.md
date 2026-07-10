# AI Fake News Detection Platform

> Detect Fake News using State-of-the-Art Transformer Models with Explainable AI (XAI)

## Project Overview

The **AI Fake News Detection Platform** is a production-ready web application that classifies news articles as **Real** or **Fake** using Transformer-based NLP models. It also explains predictions using Explainable AI (SHAP/LIME).

## Objectives

- Detect fake news accurately.
- Explain predictions with XAI.
- Build a scalable full-stack AI application.
- Deploy to cloud platforms.

## Tech Stack

### Frontend
- Next.js 15
- React
- TypeScript
- Tailwind CSS
- Shadcn/UI
- Framer Motion

### Backend
- FastAPI
- Python 3.12
- Uvicorn
- Pydantic

### AI
- HuggingFace Transformers
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- SpaCy
- NLTK

### Explainability
- SHAP
- LIME
- Captum

### Database
- PostgreSQL (Supabase) or MongoDB

---

# Recommended Datasets

## Primary
### WELFake Dataset
- 72,000+ news articles
- Columns: Title, Text, Label
- Label: 0 = Fake, 1 = Real

https://www.kaggle.com/datasets/studymart/welfake-dataset-for-fake-news

## Secondary
### FakeNewsNet

https://www.kaggle.com/datasets/mdepak/fakenewsnet

## Benchmark
### LIAR Dataset

https://aclanthology.org/P17-2067/

---

# AI Pipeline

```text
News Article
      ↓
Preprocessing
      ↓
Tokenization
      ↓
Transformer Model
      ↓
Prediction
      ↓
Confidence Score
      ↓
Explainable AI
      ↓
Final Result
```

## Preprocessing
- Remove HTML
- Remove URLs
- Remove Emojis
- Lowercase
- Remove Punctuation
- Remove Extra Spaces
- Lemmatization
- Tokenization

## Recommended Models
- DeBERTa-v3 ⭐⭐⭐⭐⭐
- RoBERTa
- BERT
- DistilBERT
- Longformer

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Features
- Fake/Real Prediction
- Confidence Score
- Explainable AI
- Keyword Extraction
- Topic Classification
- Named Entity Recognition
- Sentiment Analysis
- AI Summary
- Related Articles
- User Dashboard
- Admin Dashboard

## REST API

- POST /predict
- POST /explain
- POST /summary
- POST /keywords
- POST /topics
- GET /metrics

## Folder Structure

```text
AI-Fake-News-Detection/
├── frontend/
├── backend/
├── datasets/
├── models/
├── training/
├── preprocessing/
├── explainability/
├── api/
├── tests/
├── docs/
├── README.md
└── requirements.txt
```

## Future Enhancements

- Browser Extension
- OCR Support
- Voice Input
- Multilingual Detection
- RAG with Trusted Sources
- Live News Monitoring
- Docker & CI/CD

## License

MIT License
