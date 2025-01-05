import torch
from cleantext import clean
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def preprocess(text):
    """
    Preprocess input text: removes unnecessary line breaks, normalizes punctuation, and cleans text.
    """
    # Normalize line breaks
    text = text.replace("\n", " ").replace("\\n", " ").strip()
    
    # Normalize text using cleantext library
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        lang="en"
    )
    return text

def detect(input_text, tokenizer, model, device="cpu", th=-3.08583984375):
    """
    Perform inference using the provided model and tokenizer.
    """
    try:
        # Tokenize and prepare the input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        # Perform inference
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute probabilities and determine the predicted class
        probabilities = torch.softmax(logits, dim=1).squeeze()
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

        # Map the predicted class to the label
        label_map = {1: "machine-generated", 0: "human-written"}
        label = label_map.get(predicted_class, "unknown")
        logprobs = probabilities.tolist()
        logprobs.reverse()
        return {"label": label, "confidence": confidence, "logprobs": logprobs}

    except Exception as e:
        return {"error": f"An error occurred during detection: {str(e)}"}
