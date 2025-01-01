import torch
from cleantext import clean
from transformers import AutoTokenizer, AutoModelForSequenceClassification,LogitsProcessorList
from backend.watermark.extended_watermark_processor import WatermarkLogitsProcessor

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

def detect(input_text, watermark_detector, device="cpu"):
    """
    Perform inference using the provided model and tokenizer.
    """
    
    score_dict = watermark_detector.detect(input_text) 

    return {
        "label": "machine-generated" if score_dict['prediction'] else "human-written",
        "confidence": round(1 - score_dict['p_value'], 4),  # Rounded for better readability
        "num_tokens_scored": score_dict.get('num_tokens_scored', None),
        "num_green_tokens": score_dict.get('num_green_tokens', None),
        "green_fraction": round(score_dict.get('green_fraction', 0), 4),
        "z_score": round(score_dict.get('z_score', 0), 4),
        "p_value": round(score_dict.get('p_value', 0), 8),  # Rounded for precision in display
        "detection_threshold": 2.0,  # The z_threshold used in the detection logic
    }
    
    
    

def watermarkedtext(input_text, tokenizer, model, device="cpu"):
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                gamma=0.25,
                                                delta=2.0,
                                                seeding_scheme="selfhash") #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
    tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)

    output_tokens = model.generate(**tokenized_input,
                                logits_processor=LogitsProcessorList([watermark_processor]))

    # if decoder only model, then we need to isolate the
    # newly generated tokens as only those are watermarked, the input/prompt is not
    output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    return output_text
