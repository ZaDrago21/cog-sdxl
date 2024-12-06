from constants import * # constants.py
DEFAULT_VAE_NAME = BAKEDIN_VAE_LABEL if DEFAULT_VAE_NAME is None else DEFAULT_VAE_NAME

assert len(MODELS) > 0, f"You don't have any model under \"{MODELS_DIR_PATH}\", please put at least 1 model in there!"
assert DEFAULT_VAE_NAME == BAKEDIN_VAE_LABEL or DEFAULT_VAE_NAME in VAE_NAMES, f"You have set a default VAE but it's not found under \"{VAES_DIR_PATH}\"!"
assert DEFAULT_CLIP_SKIP >= 1, f"CLIP skip must be at least 1 (which is no skip), this is the behavior in A1111 so it's aligned to it!"

from cog import BasePredictor, Input, Path
import utils # utils.py
import os
import random
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoencoderKL
from schedulers import SDXLCompatibleSchedulers # schedulers.py
from loras import SDXLMultiLoRAHandler # loras.py
from compel import Compel, ReturnedEmbeddingsType
import re
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)

from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import (
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)

from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)

def parse_generic_weights(text):
    pattern = r'\(([^:]+):\s*([^)]+?)\s*\)'
    matches = re.finditer(pattern, text)
    results = []
    for match in matches:
        original_match = match.group(0)  
        content = match.group(1).strip() 
        number_str = match.group(2).strip()
        try:
            number = float(number_str)  
        except ValueError:
            number = None
        results.append((original_match, content, number))
    for item in results:
        text = text.replace(item[0], "("+item[1]+")"+str(item[2]))
    return text


def parse_nai_weights(text):
    pattern = r'(\{+)(\w+)(\}+)|(\[+)([^\]]+)(\]+)'
    matches = re.finditer(pattern, text)
    results = {}
    for match in matches:
        original_match = match.group(0)
        if match.group(1):
            num_open_braces = len(match.group(1))
            num_close_braces = len(match.group(3))
            num_braces = min(num_open_braces, num_close_braces)
            name = match.group(2)
            results[name] = (original_match, name, num_braces)
        elif match.group(4):
            num_open_brackets = len(match.group(4))
            num_close_brackets = len(match.group(6))
            num_braces = min(num_open_brackets, num_close_brackets)
            name = match.group(5)
            results[name] = (original_match, name, num_braces)     
    for item, result in results.items():
        if result[0][0] == "[":
            calculated_value = round(1.0 / (1.05**result[2]), 4)
        else:
            calculated_value = round(1.05**result[2], 4)
        text = text.replace(result[0], "("+result[1]+")"+str(calculated_value))
    return text
    
def parse_webui_weights(text):
    pattern = r'(?<!_|\\)(\(+)(\w+)(\)+)(?!_|\\|\d+)|(\[+)([^\]]+)(\]+)(?!_|\\|\d+)'
    matches = re.finditer(pattern, text)
    results = {}
    for match in matches:
        original_match = match.group(0)
        if match.group(1):
            num_open_braces = len(match.group(1))
            num_close_braces = len(match.group(3))
            num_braces = min(num_open_braces, num_close_braces)
            name = match.group(2)
            results[name] = (original_match, name, num_braces)
        elif match.group(4):
            num_open_brackets = len(match.group(4))
            num_close_brackets = len(match.group(6))
            num_braces = min(num_open_brackets, num_close_brackets)
            name = match.group(5)
            results[name] = (original_match, name, num_braces)     
    for item, result in results.items():
        if result[0][0] == "[":
            calculated_value = round(1.0 / (1.1**result[2]), 4)
        else:
            calculated_value = round(1.1**result[2], 4)
        text = text.replace(result[0], "("+result[1]+")"+str(calculated_value))
    return text

def parse_weights(text):
    if "{" in text: # preffer to use nai weighting system first. {1.05} [0.95]
        text = parse_nai_weights(text)
        text = parse_webui_weights(text)
        text = parse_generic_weights(text)
    else: # preffer to use webui weights (1.1) [0.9]
        text = parse_webui_weights(text) 
        text = parse_nai_weights(text)
        text = parse_generic_weights(text)
    return text



# Cog will only run this class in a single thread.
class Predictor(BasePredictor):

    def setup(self):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        self.pipelines = SDXLMultiPipelineHandler(MODELS, VAES_DIR_PATH, VAE_NAMES, TEXTUAL_INVERSION_PATHS, TORCH_DTYPE, CPU_OFFLOAD_INACTIVE_MODELS)
        self.loras = SDXLMultiLoRAHandler()
        os.makedirs("tmp", exist_ok=True)

    def predict(
        self,
        model: str = Input(description="The model to use", default=DEFAULT_MODEL, choices=MODEL_NAMES),
        vae: str = Input(
            description="The VAE to use",
            default=DEFAULT_VAE_NAME,
            choices=list(dict.fromkeys([DEFAULT_VAE_NAME, BAKEDIN_VAE_LABEL] + VAE_NAMES + MODEL_NAMES)),
        ),
        prompt: str = Input(description="The prompt", default=DEFAULT_POSITIVE_PROMPT),
        image: Path = Input(description="The image for image to image or as the base for inpainting (Will be scaled then cropped to the set width and height)", default=None),
        mask: Path = Input(description="The mask for inpainting, white areas will be modified and black preserved (Will be scaled then cropped to the set width and height)", default=None),
        loras: str = Input(
            description="The LoRAs to use, must be either a string with format \"URL:Strength,URL:Strength,...\" (Strength is optional, default to 1), "
                        "or a JSON list dumped as a string containing key \"url\" (Required), \"strength\" (Optional, default to 1), and \"civitai_token\" (Optional, for downloading from CivitAI) "
                        "(NOTICE: Will download the weights, might take a while if the LoRAs are huge or the download is slow, WILL CHARGE WHEN DOWNLOADING)",
            default=DEFAULT_LORA,
        ),
        negative_prompt: str = Input(description="The negative prompt (For things you don't want)", default=DEFAULT_NEGATIVE_PROMPT),
        cfg_scale: float = Input(description="CFG scale defines how much attention the model pays to the prompt when generating, set to 1 to disable", default=DEFAULT_CFG, ge=1, le=50),
        guidance_rescale: float = Input(description="The amount to rescale CFG generated noise to avoid generating overexposed images, set to 0 or 1 to disable", default=DEFAULT_RESCALE, ge=0, le=5),
        pag_scale: float = Input(description="PAG scale is similar to CFG but it literally makes the result better, it's compatible with CFG too, set to 0 to disable", default=DEFAULT_PAG, ge=0, le=50),
        clip_skip: int = Input(description="How many CLIP layers to skip, 1 is actually no skip, this is the behavior in A1111 so it's aligned to it", default=DEFAULT_CLIP_SKIP, ge=1),
        width: int = Input(description="The width of the image", default=DEFAULT_WIDTH, ge=1, le=4096),
        height: int = Input(description="The height of the image", default=DEFAULT_HEIGHT, ge=1, le=4096),
        prepend_preprompt: bool = Input(description=f"Prepend preprompt (Prompt: \"{DEFAULT_POS_PREPROMPT}\" Negative prompt: \"{DEFAULT_NEG_PREPROMPT}\")", default=True),
        scheduler: str = Input(description="The scheduler to use", default=DEFAULT_SCHEDULER, choices=SCHEDULER_NAMES),
        steps: int = Input(description="The steps when generating", default=DEFAULT_STEPS, ge=1, le=100),
        strength: float = Input(description="How much noise to add (For image to image and inpainting only, larger value indicates more noise added to the input image)", default=0.7, ge=0, le=1),
        blur_factor: float = Input(description="The factor to blur the inpainting mask for smoother transition between masked and unmasked", default=5, ge=0),
        batch_size: int = Input(description="Number of images to generate (1-4), note if you set this to 4, some high resolution gens might fail because of not enough VRAM", default=1, ge=1, le=4),
        seed: int = Input(description="The seed used when generating, set to -1 for random seed", default=-1),
        lora_scale: float = Input(description="Lora scale for all loras in weighting prompts the <lora:url:1.0> 1.0 will be ignored only lora_scale will be applied", default=1.0),
        prompt_emebding: bool = Input(description="if to enable 77+ token support by converting to embeds otherwise will use previous prompt/neg prompts.", default=False),

        
    ) -> list[Path]:
        if prompt == "__ignore__":
            return []
        if prepend_preprompt:
            prompt = DEFAULT_POS_PREPROMPT + prompt
            negative_prompt = DEFAULT_NEG_PREPROMPT + negative_prompt
         prompt = parse_weights(prompt)    
         negative_prompt = parse_weights(negative_prompt)
        
            
            
        gen_kwargs = {
            "guidance_scale": cfg_scale, "guidance_rescale": guidance_rescale,
            "pag_scale": pag_scale, "clip_skip": clip_skip - 1, "num_inference_steps": steps, "num_images_per_prompt": batch_size,
        }
        pipeline = self.pipelines.get_pipeline(model, None if vae == BAKEDIN_VAE_LABEL else vae, scheduler)

        try:
            self.loras.process(loras, pipeline)    
            
            if prompt_emebding:
               # Initialize Compel with both sets of tokenizers and text encoders
               compel = Compel(
                   tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                   text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                   device='cuda',
                   returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
                   requires_pooled=[False, True],
                   truncate_long_prompts=False
               )
               
               conditioning, pooled = compel([prompt, negative_prompt])


               gen_kwargs["prompt_embeds"] = conditioning[0:1]
               gen_kwargs["negative_prompt_embeds"] = conditioning[1:2]
               gen_kwargs["pooled_prompt_embeds"] = pooled[0:1]
               gen_kwargs["negative_pooled_prompt_embeds"] = pooled[1:2]
            else:
                gen_kwargs["prompt"] = prompt
                gen_kwargs["negative_prompt"] = negative_prompt
            
            
            if image:
                gen_kwargs["image"] = utils.scale_and_crop(image, width, height)
                gen_kwargs["strength"] = strength
                if mask:
                    # inpainting
                    mask_img = utils.scale_and_crop(mask, width, height)
                    pipeline = AutoPipelineForInpainting.from_pipe(pipeline)
                    mask_img = pipeline.mask_processor.blur(mask_img, blur_factor)
                    gen_kwargs["mask_image"] = mask_img
                    gen_kwargs["width"] = width
                    gen_kwargs["height"] = height
                    print("Using inpainting mode.")
                else:
                    # img2img
                    pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
                    print("Using image to image mode.")
            else:
                if mask:
                    raise ValueError("You must upload a base image for inpainting mode.")
                # txt2img
                gen_kwargs["width"] = width
                gen_kwargs["height"] = height
                print("Using text to image mode.")
            if seed == -1:
                seed = random.randint(0, 2147483647)
            gen_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(seed)
            print("Using seed:", seed)
            imgs = pipeline(**gen_kwargs).images

            image_paths = []
            for index, img in enumerate(imgs):
                img_file_path = f"tmp/{index}.png"
                img.save(img_file_path, optimize=True, compress_level=9)
                image_paths.append(Path(img_file_path))
            return image_paths
        finally:
            pipeline.unload_lora_weights()

class SDXLMultiPipelineHandler:

    def __init__(self, model_name_obj_dict, vaes_dir_path, vae_names, textual_inversion_paths, torch_dtype, cpu_offload_inactive_models):
        self.model_name_obj_dict = model_name_obj_dict
        self.model_pipeline_dict = {} # Key = Model's name(str), Value = StableDiffusionXLPAGPipeline instance.
        self.vaes_dir_path = vaes_dir_path
        self.vae_obj_dict = {vae_name: None for vae_name in vae_names} # Key = VAE's name(str), Value = AutoencoderKL instance.
        self.textual_inversion_paths = textual_inversion_paths
        self.torch_dtype = torch_dtype
        self.cpu_offload_inactive_models = cpu_offload_inactive_models

        self._load_all_vaes() # Must load VAEs before models.
        self._load_all_models()

        self.activated_model = None

    def get_pipeline(self, model_name, vae_name, scheduler_name):
        # __init__ function guarantees all models and VAEs to be loaded.
        pipeline = self.model_pipeline_dict.get(model_name)
        if pipeline is None:
            raise ValueError(f"Model \"{model_name}\" not found.")

        vae_name = model_name if vae_name is None else vae_name
        vae = self.vae_obj_dict.get(vae_name)
        if vae is None:
            raise ValueError(f"VAE \"{vae_name}\" not found.")

        if model_name != self.activated_model:
            if self.activated_model is not None:
                prev_activated_pipeline = self.model_pipeline_dict[self.activated_model]
                prev_activated_pipeline.vae = None
                if self.cpu_offload_inactive_models:
                    prev_activated_pipeline.to("cpu")
            self.activated_model = model_name

        pipeline.to("cuda")
        pipeline.vae = vae
        pipeline.scheduler = SDXLCompatibleSchedulers.create_instance(scheduler_name)
        return pipeline

    # Load all VAEs to GPU(CUDA).
    def _load_all_vaes(self):
        for vae_name in self.vae_obj_dict:
            self.vae_obj_dict[vae_name] = self._load_vae(vae_name)

    # Load a VAE to GPU(CUDA).
    def _load_vae(self, vae_name):
        vae = AutoencoderKL.from_pretrained(os.path.join(self.vaes_dir_path, vae_name), torch_dtype=self.torch_dtype)
        vae.enable_slicing()
        vae.enable_tiling()
        vae.to("cuda")
        return vae

    # Load all models to CPU when CPU offload, to GPU(CUDA) when not CPU offload.
    def _load_all_models(self):
        clip_l_list, clip_g_list, activation_token_list = utils.get_textual_inversions(self.textual_inversion_paths)
        for model_name, model_for_loading in self.model_name_obj_dict.items():
            pipeline = self._load_model(model_name, model_for_loading, clip_l_list, clip_g_list, activation_token_list)
            if not self.cpu_offload_inactive_models:
                pipeline.to("cuda")
            self.model_pipeline_dict[model_name] = pipeline

    # Load a model to CPU.
    def _load_model(self, model_name, model_for_loading, clip_l_list, clip_g_list, activation_token_list):
        pipeline = AutoPipelineForText2Image.from_pipe(model_for_loading.load(torch_dtype=self.torch_dtype, add_watermarker=False), enable_pag=True)
        utils.apply_textual_inversions_to_sdxl_pipeline(pipeline, clip_l_list, clip_g_list, activation_token_list)
        vae = pipeline.vae
        pipeline.vae = None
        vae.enable_slicing()
        vae.enable_tiling()
        vae.to("cuda")
        self.vae_obj_dict[model_name] = vae
        return pipeline
