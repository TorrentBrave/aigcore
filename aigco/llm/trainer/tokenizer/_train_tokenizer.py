import os
import json
import aigco
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from dotenv import load_dotenv

logger = aigco.logger(name="train_tokenizer")
logger.print(f"{logger.name = }")

load_dotenv()

PRE_DATA = os.getenv("PRE_DATA_PATH")
TOKENIZER_DIR = ""
VOCAB_SIZE = 6400


def get_texts(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break
            data = json.loads(line)
            yield data["text"]


def train_tokenizer(data_path, tokenizer_dir, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train(files=[data_path], vocab_size=VOCAB_SIZE, min_frequency=2)
    tokenizer.save(TOKENIZER_DIR)
