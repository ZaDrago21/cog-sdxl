import os
import requests
import torch
from schedulers import SDXLCompatibleSchedulers # schedulers.py

TORCH_DTYPE = torch.bfloat16
REQUESTS_GLOBAL_SESSION = requests.Session()

# When set to True, the script will offload inactive models to CPU, this will add 3 seconds of overhead when switching models.
# When set to False, the script won't offload inactive models to CPU, all models will be in GPU which means they will use VRAM,
# but it should make switching models way faster because the script won't need to move the models between RAM and VRAM.
CPU_OFFLOAD_INACTIVE_MODELS = False

MODELS_DIR_PATH = "models"
MODELS_REMOTE_CACHE_PATH = "__models_remote_cache__"
import finders # finders.py | Need to import finders here because MODELS_REMOTE_CACHE_PATH must be set before model_for_loading.py imports constants.py (This script).
print("--- constants.py: Calling find_models ---") # Debug
MODELS = finders.find_models(MODELS_DIR_PATH)
print(f"--- constants.py: Found MODELS: {MODELS} ---") # Debug
MODEL_NAMES = list(MODELS)

VAES_DIR_PATH = "vaes"
print("--- constants.py: Calling find_vaes ---") # Debug
VAE_NAMES = finders.find_vaes(VAES_DIR_PATH)
print(f"--- constants.py: Found VAE_NAMES: {VAE_NAMES} ---") # Debug
VAE_NAMES.sort()

LORAS_DIR_PATH = "loras"
MAX_LORA_CACHE_BYTES = 137438953472 # 128 GB.

print("--- constants.py: Calling find_textual_inversions ---") # Debug
TEXTUAL_INVERSION_PATHS = finders.find_textual_inversions("textual_inversions")
print(f"--- constants.py: Found TEXTUAL_INVERSION_PATHS: {TEXTUAL_INVERSION_PATHS} ---") # Debug

SCHEDULER_NAMES = SDXLCompatibleSchedulers.get_names()

DEFAULT_MODEL = MODEL_NAMES[0] if len(MODEL_NAMES) else None
print(f"--- constants.py: DEFAULT_MODEL set to: {DEFAULT_MODEL} ---") # Debug

DEFAULT_VAE_NAME = None
BAKEDIN_VAE_LABEL = "default"

DEFAULT_LORA = None

DEFAULT_POS_PREPROMPT = "score_9, score_8_up, score_7_up, "
DEFAULT_NEG_PREPROMPT = "score_4, score_3, score_2, score_1, worst quality, bad hands, bad feet, "

DEFAULT_POSITIVE_PROMPT = "1girl"
DEFAULT_NEGATIVE_PROMPT = "animal, cat, dog, big breasts"

DEFAULT_CFG = 5
DEFAULT_RESCALE = 0.5
DEFAULT_PAG = 3

# CLIP skip is calculated in the same way as AUTOMATIC1111 and CivitAI.
DEFAULT_CLIP_SKIP = 1

DEFAULT_WIDTH = 1184
DEFAULT_HEIGHT = 864

DEFAULT_SCHEDULER = SCHEDULER_NAMES[0]
DEFAULT_STEPS = 35

# Upscaler constants
UPSCALE_DIR_PATH = "upscalers"
UPSCALE_MODELS = []
print(f"--- constants.py: Checking UPSCALE_DIR_PATH: {UPSCALE_DIR_PATH} ---") # Debug
if os.path.exists(UPSCALE_DIR_PATH):
    print(f"--- constants.py: UPSCALE_DIR_PATH exists. Listing contents...") # Debug
    for fname in os.listdir(UPSCALE_DIR_PATH):
        if fname.lower().endswith(".pth"):
            UPSCALE_MODELS.append(os.path.join(UPSCALE_DIR_PATH, fname))
else:
    print(f"--- constants.py: UPSCALE_DIR_PATH does not exist.") # Debug
print(f"--- constants.py: Found UPSCALE_MODELS: {UPSCALE_MODELS} ---") # Debug
UPSCALE_MODELS.sort()  # Sort alphabetically, if needed.
DEFAULT_UPSCALE_MODEL = UPSCALE_MODELS[0] if UPSCALE_MODELS else None
print(f"--- constants.py: DEFAULT_UPSCALE_MODEL set to: {DEFAULT_UPSCALE_MODEL} ---") # Debug
