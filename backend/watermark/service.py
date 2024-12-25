from argparse import Namespace
from backend.watermark.utils import preprocess, detect, watermarkedtext
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from backend.watermark.models import DetectionRequest, DetectionResponse, Error
from langdetect import detect as lang_detect, DetectorFactory, LangDetectException
from backend.watermark.helpers import load_model
# Conditionally import torch if available in the environment
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
config = {
    'run_gradio': True,
    'demo_public': False,
    'model_name_or_path': 'facebook/opt-125m',
    'prompt_max_length': None,
    'max_new_tokens': 200,
    'generation_seed': 123,
    'use_sampling': True,
    'sampling_temp': 0.7,
    'n_beams': 1,
    'use_gpu': True,
    'seeding_scheme': 'simple_1',
    'gamma': 0.25,
    'delta': 2.0,
    'normalizers': '',
    'ignore_repeated_bigrams': False,
    'detection_z_threshold': 4.0,
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'load_fp16': False,
}

class InferenceEngine:
    def __init__(self):
        
        
        #args = parse_args()
        args = Namespace(**config)  # Convert the config dictionary to a Namespace
        args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
        model, tokenizer, device = load_model(args)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


    def predict(self, detection_request: DetectionRequest) -> DetectionResponse:
        """
        Run inference on the input text and return a DetectionResponse.
        """
        if not torch_available:
            raise ImportError("torch is not available in the environment.")
        inputs = preprocess(detection_request.text)
        outputs = detect(inputs, self.tokenizer, self.model, self.device)
        watermarked_output=watermarkedtext(inputs, self.tokenizer, self.model, self.device)
        return DetectionResponse(
            text=detection_request.text,
            label=outputs["label"],
            confidence=outputs["confidence"],
        )