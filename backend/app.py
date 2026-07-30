import os
import re
import json
import pickle
import sys
import urllib.request
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path so we can import preprocessing and explainability
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cleaner import TextCleaner
from explainability.explainer import PerturbationExplainer

# Load .env manually
def load_env():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

load_env()

def call_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        raise RuntimeError(f"Gemini API error: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="AI Fake News Detection Platform API",
    description="Backend API for predicting and explaining real/fake news.",
    version="1.0.0"
)

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP elements
cleaner = TextCleaner()
explainer = PerturbationExplainer()

# Global variables for models
model = None
vectorizer = None
transformer_pipeline = None

# Topic keywords mapping
TOPIC_KEYWORDS = {
    "politics": ["government", "law", "president", "election", "debate", "senate", "congress", "minister", "state", "federal", "policy", "vote", "democrat", "republican", "bill", "campaign", "political", "court", "judge", "obama", "trump", "biden", "white house"],
    "technology": ["computer", "software", "ai", "robot", "solar", "satellite", "internet", "phone", "tech", "digital", "app", "space", "launch", "astronomer", "nasa", "cybersecurity", "hacker", "database", "device", "engineering", "innovative"],
    "business": ["market", "trade", "economy", "tariff", "bank", "tax", "company", "ceo", "stock", "wall street", "dollar", "price", "revenue", "growth", "rate", "consumer", "financial", "corporate", "industry", "investment", "business"],
    "science": ["energy", "climate", "warming", "carbon", "research", "scientist", "study", "university", "biology", "chemistry", "physics", "fossil", "earth", "laboratory", "species", "genetics", "nature", "cell", "quantum", "discover"],
    "health": ["vaccine", "virus", "health", "flu", "doctor", "hospital", "case", "infection", "clinical", "medicine", "drug", "treatment", "disease", "medical", "patient", "wellness", "mental", "illness", "outbreak"],
    "sports": ["player", "game", "win", "score", "champion", "coach", "stadium", "basketball", "football", "soccer", "olympics", "match", "tournament", "league", "team", "athlete", "tennis", "cup", "gold medal"]
}

# Request/Response schemas
class PredictRequest(BaseModel):
    text: str
    model_type: str = "ml"  # "ml", "transformer", or "gemini"

class ExplainRequest(BaseModel):
    text: str
    model_type: str = "ml"

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_models():
    global model, vectorizer
    # Load Scikit-Learn model
    try:
        model_path = os.path.join("models", "model.pkl")
        vectorizer_path = os.path.join("models", "vectorizer.pkl")
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)
            print("Successfully loaded local ML model and vectorizer.")
        else:
            print("Warning: Local ML model files not found. Run training script first.")
    except Exception as e:
        print(f"Error loading local ML model: {e}")

def get_transformer_pipeline():
    global transformer_pipeline
    if transformer_pipeline is None:
        try:
            print("Loading HuggingFace transformer model...")
            from transformers import pipeline
            # Fine-tuned RoBERTa model for fake-news classification
            # Labels: "FAKE" (label 0) and "TRUE" (label 1 = real news)
            transformer_pipeline = pipeline(
                "text-classification",
                model="hamzab/roberta-fake-news-classification",
                device=-1,        # CPU; change to 0 for GPU
                truncation=True,
                max_length=512,
            )
            print("Successfully loaded HuggingFace transformer model.")
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load transformer model: {str(e)}")
    return transformer_pipeline

# Extract Summary Helper (Extractive TF-IDF)
def extract_summary(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= max_sentences:
        return text
    
    cleaned_sentences = [cleaner.clean(s) for s in sentences]
    
    # Calculate word frequency
    word_freq = {}
    for cs in cleaned_sentences:
        for word in cs.split():
            word_freq[word] = word_freq.get(word, 0) + 1
            
    if not word_freq:
        return " ".join(sentences[:max_sentences])
        
    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] /= max_freq
        
    sentence_scores = {}
    for i, cs in enumerate(cleaned_sentences):
        score = 0
        words = cs.split()
        for word in words:
            score += word_freq.get(word, 0)
        sentence_scores[i] = score / max(1, len(words))
        
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
    top_indices.sort()
    
    return " ".join([sentences[idx] for idx in top_indices])

# Topic Classification Helper
def classify_topic(text: str) -> str:
    cleaned = text.lower()
    scores = {topic: 0 for topic in TOPIC_KEYWORDS}
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            scores[topic] += len(re.findall(r'\b' + re.escape(kw) + r'\b', cleaned))
            
    max_topic = max(scores, key=scores.get)
    if scores[max_topic] == 0:
        return "General"
    return max_topic.capitalize()

# Sentiment analysis helper
def analyze_sentiment(text: str) -> Dict[str, Any]:
    # Very simple lexicon-based sentiment analysis
    pos_words = {"good", "great", "excellent", "positive", "successful", "scientific", "breakthrough", "achievement", "benefit", "progress", "growth", "bipartisan", "support"}
    neg_words = {"bad", "worse", "terrible", "negative", "crisis", "scandal", "fraud", "conspiracy", "exposed", "hiding", "fake", "shocking", "corrupt", "illegal", "manipulate"}
    
    cleaned = cleaner.clean(text).split()
    pos_count = sum(1 for w in cleaned if w in pos_words)
    neg_count = sum(1 for w in cleaned if w in neg_words)
    
    total = pos_count + neg_count
    if total == 0:
        return {"sentiment": "NEUTRAL", "pos": 0.33, "neg": 0.33, "neu": 0.34}
        
    pos_pct = pos_count / total
    neg_pct = neg_count / total
    
    if pos_pct > 0.6:
        sentiment = "POSITIVE"
    elif neg_pct > 0.6:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
        
    return {
        "sentiment": sentiment,
        "pos": float(pos_pct),
        "neg": float(neg_pct),
        "neu": float(1.0 - pos_pct - neg_pct)
    }

# Model probabilities functions for the explainer
def get_ml_fake_probs(texts: List[str]) -> List[float]:
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Local ML model is not loaded. Train the model first.")
    cleaned_texts = [cleaner.clean(t) for t in texts]
    vecs = vectorizer.transform(cleaned_texts)
    # class 0 is FAKE, class 1 is REAL
    return list(model.predict_proba(vecs)[:, 0])

def get_transformer_fake_probs(texts: List[str]) -> List[float]:
    pipe = get_transformer_pipeline()
    truncated_texts = [t[:1500] for t in texts]
    results = pipe(truncated_texts)
    probs = []
    for res in results:
        label = res['label']   # "FAKE" or "TRUE"
        score = float(res['score'])
        # score is confidence for the predicted label
        if label == "FAKE":
            probs.append(score)
        else:
            probs.append(1.0 - score)
    return probs

# API Endpoints
@app.post("/api/predict")
def predict(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
        
    if request.model_type == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="Gemini API key is not configured.")
        
        prompt = (
            "Analyze the following news article and determine if it is REAL or FAKE. "
            "Respond with a JSON object containing: "
            "'prediction' (either 'REAL' or 'FAKE') and "
            "'confidence' (a float between 0 and 1 representing your confidence score). "
            "Respond with ONLY the JSON object. Do not include markdown formatting or backticks.\n\n"
            f"Text:\n{text}"
        )
        response_text = call_gemini(prompt)
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            data = json.loads(response_text)
        
        pred = data.get("prediction", "REAL").upper()
        conf = float(data.get("confidence", 0.85))
        
        return {
            "prediction": pred,
            "confidence": conf,
            "model_used": "Google Gemini 2.5 Flash (LLM)"
        }
        
    elif request.model_type == "ml":
        if model is None or vectorizer is None:
            raise HTTPException(status_code=500, detail="Local ML model is not trained. Please train it first.")
        
        # Predict
        cleaned = cleaner.clean(text)
        vec = vectorizer.transform([cleaned])
        pred_class = model.predict(vec)[0] # 0 = FAKE, 1 = REAL
        probs = model.predict_proba(vec)[0]
        
        label = "REAL" if pred_class == 1 else "FAKE"
        confidence = probs[1] if pred_class == 1 else probs[0]
        
        return {
            "prediction": label,
            "confidence": float(confidence),
            "model_used": "Local ML Model (TF-IDF + Logistic Regression)"
        }
        
    elif request.model_type == "transformer":
        pipe = get_transformer_pipeline()
        truncated = text[:1500]
        res = pipe(truncated)[0]
        raw_label = res["label"]   # "FAKE" or "TRUE"
        score = float(res["score"])
        label = "FAKE" if raw_label == "FAKE" else "REAL"

        return {
            "prediction": label,
            "confidence": score,
            "model_used": "Transformer Model (hamzab/roberta-fake-news-classification)"
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Must be 'ml', 'transformer', or 'gemini'.")

@app.post("/api/explain")
def explain(request: ExplainRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
        
    # We explain using the perturbation explainer
    if request.model_type == "ml":
        attributions = explainer.explain(text, get_ml_fake_probs)
    elif request.model_type == "transformer":
        attributions = explainer.explain(text, get_transformer_fake_probs)
    elif request.model_type == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="Gemini API key is not configured.")
        
        prompt = (
            "Given the following news article, select up to 10 key words/phrases that contribute to "
            "determining if it is REAL or FAKE. For each word/phrase, assign an attribution score between -1.0 and 1.0 "
            "(negative scores indicate FAKE news, positive scores indicate REAL news). "
            "Respond with ONLY a JSON array of objects, where each object has keys 'token' (the word/phrase) and 'score' (the float attribution). "
            "Do not include markdown formatting or backticks.\n\n"
            f"Article:\n{text}"
        )
        response_text = call_gemini(prompt)
        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if json_match:
            attributions = json.loads(json_match.group(0))
        else:
            attributions = json.loads(response_text)
        
        formatted_attributions = []
        for item in attributions:
            if isinstance(item, dict) and "token" in item and "score" in item:
                formatted_attributions.append({
                    "token": str(item["token"]),
                    "score": float(item["score"])
                })
        return {"attributions": formatted_attributions, "model_type": "gemini"}
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Must be 'ml', 'transformer', or 'gemini'.")
        
    return {
        "attributions": attributions,
        "model_type": request.model_type
    }

@app.post("/api/summary")
def summary(request: TextRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            prompt = f"Summarize the following news article in 2-3 sentences. Keep it objective:\n\n{request.text}"
            summary_text = call_gemini(prompt)
            return {"summary": summary_text}
        except Exception as e:
            print(f"Fallback summary due to Gemini error: {e}")
            
    return {"summary": extract_summary(request.text)}

@app.post("/api/keywords")
def keywords(request: TextRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            prompt = (
                "Extract the top 5 to 8 keywords/phrases from the following news article. "
                "Respond with a simple JSON list of strings. Do not include markdown formatting or backticks.\n\n"
                f"Article:\n{request.text}"
            )
            response_text = call_gemini(prompt)
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                kws = json.loads(json_match.group(0))
            else:
                kws = json.loads(response_text)
            return {"keywords": [str(kw) for kw in kws]}
        except Exception as e:
            print(f"Fallback keywords due to Gemini error: {e}")

    cleaned = cleaner.clean(request.text)
    words = [w for w in cleaned.split() if len(w) > 2]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:6]
    return {"keywords": [kw[0] for kw in sorted_keywords]}

@app.post("/api/topics")
def topics(request: TextRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            prompt = (
                "Classify the main topic of the following news article. "
                "Respond with only a single word from: Politics, Technology, Health, Business, Science, Sports, or General.\n\n"
                f"Article:\n{request.text}"
            )
            topic = call_gemini(prompt).strip().strip(".*").strip().capitalize()
            allowed = {"Politics", "Technology", "Health", "Business", "Science", "Sports", "General"}
            if topic in allowed:
                return {"topic": topic}
            for a in allowed:
                if a.lower() in topic.lower():
                    return {"topic": a}
        except Exception as e:
            print(f"Fallback topic due to Gemini error: {e}")
            
    return {"topic": classify_topic(request.text)}

@app.post("/api/sentiment")
def sentiment(request: TextRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            prompt = (
                "Analyze the sentiment of the following news article. "
                "Respond with a JSON object containing: "
                "'sentiment' (either 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'), "
                "'pos' (probability between 0 and 1), "
                "'neg' (probability between 0 and 1), and "
                "'neu' (probability between 0 and 1). "
                "Make sure pos + neg + neu sums to 1.0. "
                "Respond with ONLY the JSON object. Do not include markdown formatting or backticks.\n\n"
                f"Article:\n{request.text}"
            )
            response_text = call_gemini(prompt)
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = json.loads(response_text)
            
            return {
                "sentiment": data.get("sentiment", "NEUTRAL").upper(),
                "pos": float(data.get("pos", 0.33)),
                "neg": float(data.get("neg", 0.33)),
                "neu": float(data.get("neu", 0.34))
            }
        except Exception as e:
            print(f"Fallback sentiment due to Gemini error: {e}")
            
    return analyze_sentiment(request.text)

@app.get("/api/metrics")
def get_metrics():
    metrics_path = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    else:
        raise HTTPException(
            status_code=404,
            detail="Model metrics not found. Run the training script first: python training/train.py"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
