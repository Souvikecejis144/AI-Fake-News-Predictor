from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import sys
import os
import pickle

import urllib.request
import json
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.text_preprocessor import TextPreprocessor
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

app = FastAPI(
    title="AI Fake News Detection API",
    description="API for detecting fake news using Transformer models with Explainable AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
preprocessor = TextPreprocessor()
explainer = PerturbationExplainer()
model = None
vectorizer = None

# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    model_type: Optional[str] = "transformer"

class PredictionResponse(BaseModel):
    prediction: str  # "REAL" or "FAKE"
    confidence: float
    model_used: str
    warning: Optional[str] = None

class ExplanationRequest(BaseModel):
    text: str
    model_type: Optional[str] = "transformer"

class ExplanationResponse(BaseModel):
    attributions: List[Dict[str, Any]]

class SummaryRequest(BaseModel):
    text: str
    model_type: Optional[str] = "transformer"

class SummaryResponse(BaseModel):
    summary: str

class KeywordsRequest(BaseModel):
    text: str
    model_type: Optional[str] = "transformer"

class KeywordsResponse(BaseModel):
    keywords: List[str]

class TopicsRequest(BaseModel):
    text: str
    model_type: Optional[str] = "transformer"

class TopicsResponse(BaseModel):
    topic: str

class SentimentRequest(BaseModel):
    text: str
    model_type: Optional[str] = "transformer"

class SentimentResponse(BaseModel):
    sentiment: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    pos: float
    neg: float
    neu: float

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Fake News Detection API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    global model, vectorizer
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(project_root, "models", "model.pkl"), "rb") as file:
            model = pickle.load(file)
        with open(os.path.join(project_root, "models", "vectorizer.pkl"), "rb") as file:
            vectorizer = pickle.load(file)
        print("Local TF-IDF + Logistic Regression model loaded successfully!")
    except Exception as e:
        print(f"Error: Could not load the trained local model: {e}")
        import traceback
        traceback.print_exc()

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global model
    try:
        # Check text length for warning
        warning_msg = None
        if len(request.text.strip()) < 150:
            warning_msg = (
                "This text is very short. Style-based models (RoBERTa/ML) are trained on "
                "full-length articles and are less reliable for short standalone claims. "
                "Try using the Google Gemini model option for factual/real-time verification."
            )

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
                f"Text:\n{request.text}"
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
                "model_used": "Google Gemini 2.5 Flash (LLM)",
                "warning": warning_msg
            }
            
        elif request.model_type == "transformer":
            try:
                from models.model_loader import predict_with_transformer
                result = predict_with_transformer(request.text)
                return {
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "model_used": result["model_used"],
                    "warning": warning_msg
                }
            except Exception as e:
                print(f"Transformer failed, falling back to ML model: {e}")
                # Fall through to local model
 
        # Local ML model
        if model is None or vectorizer is None:
            raise HTTPException(status_code=503, detail="The trained local model is unavailable.")
 
        features = vectorizer.transform([request.text])
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        class_index = list(model.classes_).index(prediction)
        
        return {
            "prediction": "REAL" if prediction == 1 else "FAKE",
            "confidence": float(probabilities[class_index]),
            "model_used": "Local ML Model (TF-IDF + Logistic Regression)",
            "warning": warning_msg
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Explanation endpoint
@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest):
    try:
        if request.model_type == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=503, detail="Gemini API key is not configured.")
            
            prompt = (
                "Given the following news article, select up to 10 key words/phrases that contribute to "
                "determining if it is REAL or FAKE. For each word/phrase, assign an attribution score between -1.0 and 1.0 "
                "(negative scores indicate FAKE news, positive scores indicate REAL news). "
                "Respond with ONLY a JSON array of objects, where each object has keys 'token' (the word/phrase) and 'score' (the float attribution). "
                "Do not include markdown formatting or backticks.\n\n"
                f"Article:\n{request.text}"
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
            return {"attributions": formatted_attributions}
            
        elif request.model_type == "transformer":
            try:
                from models.model_loader import predict_transformer_fake_prob
                attributions = explainer.explain(request.text, predict_transformer_fake_prob)
                return {"attributions": attributions}
            except Exception as e:
                print(f"Transformer explanation failed: {e}")
                # Fall through to local model
                
        # Local model
        def predict_fn(texts: List[str]) -> List[float]:
            if model is None or vectorizer is None:
                raise HTTPException(status_code=503, detail="The trained local model is unavailable.")

            fake_class_index = list(model.classes_).index(0)
            return list(model.predict_proba(vectorizer.transform(texts))[:, fake_class_index])
        
        attributions = explainer.explain(request.text, predict_fn)
        return {"attributions": attributions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Summary endpoint
@app.post("/summary", response_model=SummaryResponse)
async def summarize(request: SummaryRequest):
    try:
        if request.model_type == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                try:
                    prompt = f"Summarize the following news article in 2-3 sentences. Keep it objective:\n\n{request.text}"
                    summary = call_gemini(prompt)
                    return {"summary": summary}
                except Exception as e:
                    print(f"Fallback summary due to Gemini error: {e}")
        
        # Simple extractive summary: first 2 sentences
        sentences = re.split(r'[.!?]+', request.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else sentences[0] if sentences else request.text[:200]
        
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keywords endpoint
@app.post("/keywords", response_model=KeywordsResponse)
async def extract_keywords(request: KeywordsRequest):
    try:
        if request.model_type == "gemini":
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
                        keywords = json.loads(json_match.group(0))
                    else:
                        keywords = json.loads(response_text)
                    return {"keywords": [str(kw) for kw in keywords]}
                except Exception as e:
                    print(f"Fallback keywords due to Gemini error: {e}")

        # Simple keyword extraction using preprocessing
        processed = preprocessor.preprocess(request.text, remove_stops=False)
        tokens = preprocessor.tokenize(processed)
        
        # Get most frequent tokens (excluding very short ones)
        from collections import Counter
        filtered_tokens = [t for t in tokens if len(t) > 3]
        counter = Counter(filtered_tokens)
        keywords = [word for word, _ in counter.most_common(10)]
        
        return {"keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Topics endpoint
@app.post("/topics", response_model=TopicsResponse)
async def classify_topics(request: TopicsRequest):
    try:
        if request.model_type == "gemini":
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

        text_lower = request.text.lower()
        
        # Simple topic classification based on keywords
        topic_keywords = {
            "Politics": ["government", "election", "president", "congress", "senate", "vote", "political", "policy"],
            "Technology": ["tech", "software", "ai", "computer", "digital", "internet", "app", "data"],
            "Health": ["health", "medical", "doctor", "hospital", "disease", "treatment", "vaccine"],
            "Business": ["business", "economy", "market", "stock", "company", "financial", "economic"],
            "Science": ["research", "study", "scientist", "discovery", "experiment", "university"],
            "Sports": ["sport", "game", "team", "player", "coach", "championship", "league"]
        }
        
        scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[topic] = score
        
        topic = max(scores.keys()) if scores else "General"
        
        return {"topic": topic}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Sentiment endpoint
@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        if request.model_type == "gemini":
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

        text_lower = request.text.lower()
        
        # Simple sentiment analysis using word lists
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "success", "happy", "positive", "best", "love"]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate", "negative", "fail", "poor", "sad"]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        total = pos_count + neg_count + 1  # +1 to avoid division by zero
        
        pos = pos_count / total
        neg = neg_count / total
        neu = 1 - (pos + neg)
        
        if pos > neg:
            sentiment = "POSITIVE"
        elif neg > pos:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return {
            "sentiment": sentiment,
            "pos": pos,
            "neg": neg,
            "neu": max(0, neu)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    import json as _json
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "metrics.json"
    )
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return _json.load(f)
    raise HTTPException(
        status_code=404,
        detail="Model metrics not found. Run the training script first: python training/train.py"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

