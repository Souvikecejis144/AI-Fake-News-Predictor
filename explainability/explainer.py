import re
from typing import List, Dict, Any, Callable

class PerturbationExplainer:
    def __init__(self):
        pass

    def explain(self, text: str, predict_fn: Callable[[List[str]], List[float]]) -> List[Dict[str, Any]]:
        """
        Explains the prediction by perturbing words in the text.
        
        Args:
            text: The input text to explain.
            predict_fn: A function that takes a list of texts and returns a list of float probabilities
                        representing the confidence of being 'FAKE' (between 0 and 1).
                        
        Returns:
            A list of dicts containing the original tokens and their attribution scores.
        """
        if not text:
            return []
            
        # Split text into tokens including whitespace and punctuation
        tokens = re.split(r'(\s+|[^\w\s\'])', text)
        tokens = [t for t in tokens if t] # filter out empty strings
        
        # Identify "word" tokens (exclude pure whitespace and punctuation)
        word_indices = []
        words_to_perturb = []
        
        for i, token in enumerate(tokens):
            # If token contains alphanumeric characters, we count it as a word to perturb
            if re.search(r'\w', token):
                word_indices.append(i)
                words_to_perturb.append(token)
                
        if not words_to_perturb:
            # Nothing to perturb, return all tokens with zero score
            return [{"token": t, "score": 0.0} for t in tokens]
            
        # Calculate baseline prediction for the full text
        baseline_prob = predict_fn([text])[0]
        
        # Prepare perturbed texts (each omitting one word)
        perturbed_texts = []
        for i in word_indices:
            perturbed_tokens = tokens.copy()
            # Replace the word at index i with an empty string
            perturbed_tokens[i] = ""
            perturbed_text = "".join(perturbed_tokens)
            perturbed_texts.append(perturbed_text)
            
        # Run batch prediction for all perturbed texts
        try:
            perturbed_probs = predict_fn(perturbed_texts)
        except Exception as e:
            print(f"Error during perturbation prediction: {e}")
            # fallback: predict one by one
            perturbed_probs = []
            for pt in perturbed_texts:
                try:
                    perturbed_probs.append(predict_fn([pt])[0])
                except Exception:
                    perturbed_probs.append(baseline_prob)
                    
        # Calculate word attribution scores
        # Attribution score = baseline_prob - perturbed_prob
        # If baseline_prob is high (e.g. 0.9 FAKE) and removing the word drops the prob to 0.7,
        # then the score is 0.9 - 0.7 = +0.2. This word contributed positively to the FAKE prediction!
        # If removing the word increases the prob to 0.95, score is 0.9 - 0.95 = -0.05.
        # This word contributed negatively (made it look more REAL).
        word_scores = {}
        for idx, word_idx in enumerate(word_indices):
            prob_without_word = perturbed_probs[idx]
            score = baseline_prob - prob_without_word
            word_scores[word_idx] = score
            
        # Reconstruct the token list with scores
        result = []
        for i, token in enumerate(tokens):
            score = word_scores.get(i, 0.0)
            result.append({
                "token": token,
                "score": float(score)
            })
            
        return result
