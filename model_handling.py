from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from utils import logger

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = f'cuda:{cuda.current_device()} : GPU' if cuda.is_available() else 'cpu'

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
logger.info(f"During this run {device} was used")


def get_model_and_tokenizer(MODEL_ID:str, MODEL_DIR:str): 
    model_config = transformers.AutoConfig.from_pretrained(
        MODEL_ID,
        cache_dir = MODEL_DIR
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        cache_dir = MODEL_DIR
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    return model, tokenizer

