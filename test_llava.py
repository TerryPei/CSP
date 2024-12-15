from transformers.utils import is_offline, logging
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
print(f"Transformers cache directory is: {cache_dir}")
