from backend.finetuned.utils import preprocess, detect
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from backend.finetuned.models import DetectionRequest, DetectionResponse, Error
from langdetect import detect as lang_detect, DetectorFactory, LangDetectException
from logging import getLogger
import json
# Conditionally import torch if available in the environment
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

logger = getLogger(__name__)

class InferenceEngine:
    def __init__(self):
        """
        Initialize the inference engine with models for different languages.
        """
        self.device = "cuda" if torch_available and torch.cuda.is_available() else "cpu"
        # Define model paths for each language
        self.model_dirs = {
            "en": "models/bert-base-multilingual-cased-finetuned-en-text-davinci-003",
            "es": "models/bert-base-multilingual-cased-finetuned-es-text-davinci-003",
            "all": "models/bert-base-multilingual-cased-finetuned-en-text-davinci-003"
            # "all": "models/roberta-large-openai-detector-finetuned-all-all"
            # Add paths for more languages as needed
        }

        # Load models and tokenizers for each language
        logger.info(f"Loading {json.dumps(self.model_dirs)}")
        self.models = {
            lang: AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
            for lang, model_dir in self.model_dirs.items()
        }
        self.tokenizers = {
            lang: AutoTokenizer.from_pretrained(model_dir)
            for lang, model_dir in self.model_dirs.items()
        }

        # Set the fallback language (all)
        self.default_lang = "all"

    def predict(self, detection_request: DetectionRequest) -> DetectionResponse:
        """
        Run inference on the input text and return a DetectionResponse.
        """
        if not torch_available:
            raise ImportError("torch is not available in the environment.")
        
        # Preprocess the text
        inputs = preprocess(detection_request.text)

        # Detect the language of the text
        try:
            detected_lang = lang_detect(inputs)
            print(f"Detected language: {detected_lang}")
        except LangDetectException:
            print("Language detection failed. Falling back to default language: English.")
            detected_lang = self.default_lang

        # Use default language model if the detected language is unsupported
        if detected_lang not in self.models:
            print(f"Unsupported language '{detected_lang}'. Falling back to default language: English.")
            detected_lang = self.default_lang

        # Route the input to the appropriate model and tokenizer
        model = self.models[detected_lang]
        tokenizer = self.tokenizers[detected_lang]

        # Perform inference
        outputs = detect(inputs, tokenizer, model, self.device)
        return DetectionResponse(
            **outputs
        )
