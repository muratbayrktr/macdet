import torch
from cleantext import clean
from transformers import AutoTokenizer, AutoModelForSequenceClassification,LogitsProcessorList
from backend.watermark.extended_watermark_processor import WatermarkLogitsProcessor
from backend.watermark.extended_watermark_processor import WatermarkDetector

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

def detect(input_text, tokenizer, model, device="cpu"):
    """
    Perform inference using the provided model and tokenizer.
    """
            
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=0.25, # should match original setting
                                            seeding_scheme="selfhash", # should match original setting
                                            device=model.device, # must match the original rng device type
                                            tokenizer=tokenizer,
                                            z_threshold=4.0,
                                            normalizers=[],
                                            ignore_repeated_ngrams=True)
    
    score_dict = watermark_detector.detect(input_text) 

    label_map = {True: "machine-generated", False: "human-written"}
    label = label_map.get(score_dict['prediction'], "unknown")
    confidence= 1-score_dict['p_value']
    return {
        "label": label,
        "confidence": confidence,
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
    print(f"watermarked:{output_text}")
    return output_text
