# Torch vision functional tensor fix
import sys
import types
from torchvision.transforms.functional import rgb_to_grayscale

# Create a module for `torchvision.transforms.functional_tensor`
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale

# Add this module to sys.modules so other imports can access it
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor


from constants import * # constants.py
DEFAULT_VAE_NAME = BAKEDIN_VAE_LABEL if DEFAULT_VAE_NAME is None else DEFAULT_VAE_NAME

try:
    assert len(MODELS) > 0, f"You don't have any model under \"{MODELS_DIR_PATH}\", please put at least 1 model in there!"
    assert DEFAULT_VAE_NAME == BAKEDIN_VAE_LABEL or DEFAULT_VAE_NAME in VAE_NAMES, f"You have set a default VAE but it's not found under \"{VAES_DIR_PATH}\"!"
    assert DEFAULT_CLIP_SKIP >= 1, f"CLIP skip must be at least 1 (which is no skip), this is the behavior in A1111 so it's aligned to it!"
except Exception as e:
    print("Assert error:", e)



from cog import BasePredictor, Input, Path
import utils # utils.py
import os
import random
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoencoderKL
from schedulers import SDXLCompatibleSchedulers # schedulers.py
from loras import SDXLMultiLoRAHandler # loras.py
from compel import Compel, ReturnedEmbeddingsType
import re
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
from diffusers import DiffusionPipeline

from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)

# Use the external module for parsing weights
from weighting import parse_weights

# Import RealESRGAN from the top
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer  # <-- Added for face restoration

import requests
from urllib.parse import urlparse

# Cog will only run this class in a single thread.
class Predictor(BasePredictor):

    def setup(self):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        self.pipelines = SDXLMultiPipelineHandler(MODELS, VAES_DIR_PATH, VAE_NAMES, TEXTUAL_INVERSION_PATHS, TORCH_DTYPE, CPU_OFFLOAD_INACTIVE_MODELS)
        self.loras = SDXLMultiLoRAHandler()
        os.makedirs("tmp", exist_ok=True)

    def predict(
        self,
        # --- Debug: Use static choices for schema validation ---
        model: str = Input(description="The model to use", default=DEFAULT_MODEL, choices=[DEFAULT_MODEL] if DEFAULT_MODEL else ["placeholder-model"]), 
        vae: str = Input(
            description="The VAE to use",
            default=DEFAULT_VAE_NAME,
            choices=[DEFAULT_VAE_NAME] if DEFAULT_VAE_NAME else ["placeholder-vae"] # Static list for debug
        ),
        # --- End Debug ---
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
        hiresfix: bool = Input(description="If to use hiresfix", default=False),
        # --- Debug: Use static choices for schema validation ---
        hiresfix_model: str = Input(
            description="URI or local path to the RealESRGAN model weights for hiresfix. Choose from available upscale models.",
            default=DEFAULT_UPSCALE_MODEL,
            choices=[DEFAULT_UPSCALE_MODEL] if DEFAULT_UPSCALE_MODEL else ["placeholder-upscaler"] # Static list for debug
        ),
        # --- End Debug ---
        hiresfix_scale: float = Input(description="The scale factor for the hiresfix model", default=4),
        face_restoration: bool = Input(description="Apply GFPGAN-based face restoration for enhanced facial details", default=False),
        gfpgan_model: str = Input(description="Path or URL to the GFPGAN model weights", default="gfpgan/GFPGANv1.4.pth"),
        compress_level: int = Input(
            description="PNG compression level (0 = no compression, 9 = maximum compression)",
            default=6,
            ge=0,
            le=9,
        ),
        optimize: bool = Input(
            description="Enable PNG optimization during saving. Disable this to skip optimization.",
            default=True,
        ),
        use_refiner: bool = Input(description="Enable SDXL Refiner model (Disables hires fix, img2img, and inpainting)", default=False),
        high_noise_frac: float = Input(description="Fraction of steps for base model when using refiner (e.g., 0.8 means base runs for 80% of steps)", default=0.8, ge=0, le=1),
    ) -> list[Path]:
        if prompt == "__ignore__":
            return []
        if prepend_preprompt:
            prompt = DEFAULT_POS_PREPROMPT + prompt
            negative_prompt = DEFAULT_NEG_PREPROMPT + negative_prompt
        prompt = parse_weights(prompt)    
        negative_prompt = parse_weights(negative_prompt)
        
        # --- Initialize GFPGANer if requested ---
        gfpganer = None
        if face_restoration:
            if not os.path.exists(gfpgan_model):
                 print(f"Warning: GFPGAN model file not found at {gfpgan_model}. Disabling face restoration.")
                 face_restoration = False # Disable if model not found
            else:
                try:
                    # Assume RealESRGAN isn't needed as a background upsampler here
                    # You might need to adjust upscale/arch based on the specific GFPGAN model if not v1.4
                    gfpganer = GFPGANer(
                        model_path=gfpgan_model,
                        upscale=1, # Typically 1 when used for restoration only
                        arch='clean', # or 'original'
                        channel_multiplier=2,
                        bg_upsampler=None 
                    )
                    print("GFPGAN face restoration is enabled and initialized.")
                except Exception as e:
                    print(f"Warning: Failed to initialize GFPGANer. Disabling face restoration. Error: {e}")
                    face_restoration = False # Disable on error
        # --- End GFPGAN Init ---

        # --- Refiner Conflict Check ---
        should_use_refiner = use_refiner
        if use_refiner:
            if hiresfix:
                print("Warning: Hires fix is enabled, disabling SDXL Refiner.")
                should_use_refiner = False
            elif image:
                print("Warning: Image input (img2img/inpainting) detected, disabling SDXL Refiner.")
                should_use_refiner = False
            elif not self.pipelines.refiner_pipeline:
                 print("Warning: Refiner requested but not loaded/available, disabling SDXL Refiner.")
                 should_use_refiner = False
        # --- End Refiner Check ---
            
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
            
            # --- Main Generation Logic ---
            if should_use_refiner:
                 print(f"Using Refiner: Base steps {int(steps * high_noise_frac)}, Refiner steps {steps - int(steps * high_noise_frac)}")
                 # Base model pass (output latents)
                 gen_kwargs["output_type"] = "latent"
                 gen_kwargs["num_inference_steps"] = int(steps * high_noise_frac)
                 latents = pipeline(**gen_kwargs).images
                 
                 # Refiner pass
                 refiner_kwargs = {
                     "prompt": prompt,
                     "negative_prompt": negative_prompt,
                     "num_inference_steps": steps, # Refiner needs total steps AFAIK, but denoising starts from high_noise_frac
                     "denoising_start": high_noise_frac, # Specify where refiner takes over
                     "image": latents, # Pass latents from base
                     "generator": gen_kwargs["generator"], # Reuse generator for consistency
                     "num_images_per_prompt": batch_size,
                 }
                 # Use embeddings if calculated earlier
                 if prompt_emebding:
                    refiner_kwargs["prompt_embeds"] = gen_kwargs["prompt_embeds"]
                    refiner_kwargs["negative_prompt_embeds"] = gen_kwargs["negative_prompt_embeds"]
                    refiner_kwargs["pooled_prompt_embeds"] = gen_kwargs["pooled_prompt_embeds"]
                    refiner_kwargs["negative_pooled_prompt_embeds"] = gen_kwargs["negative_pooled_prompt_embeds"]
                    refiner_kwargs.pop("prompt", None)
                    refiner_kwargs.pop("negative_prompt", None)
                    
                 imgs = self.pipelines.refiner_pipeline(**refiner_kwargs).images
            
            elif not hiresfix:
                 # Standard txt2img/img2img/inpainting without refiner or hires fix
                 gen_kwargs["output_type"] = "pil" # Ensure PIL output if not using refiner/latents
                 imgs = pipeline(**gen_kwargs).images
            else:
                # --- Hires Fix Logic (largely unchanged) ---
                # Step 1: Use the passed image if provided, otherwise generate a full-resolution image.
                if image is not None:
                    full_res_img = Image.open(str(image))
                    full_res_img = full_res_img.convert("RGB")
                    full_res_imgs = [full_res_img]
                    print("Using the provided image for hiresfix processing.")
                else:
                    full_res_imgs = pipeline(**gen_kwargs).images

                # Prepare the RealESRGAN (hiresfix) upsampler.
                if not os.path.exists(hiresfix_model):
                    raise FileNotFoundError(f"Upscaler weight file not found: {hiresfix_model}")
                backbone_model, netscale = get_upscale_backbone(os.path.splitext(os.path.basename(hiresfix_model))[0]) # Derive from filename
                upsampler = RealESRGANer(
                    scale=netscale,
                    model_path=hiresfix_model,
                    dni_weight=None,
                    model=backbone_model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=self.torch_dtype == torch.float16, # Set based on torch_dtype
                    gpu_id=0
                )
                
                fixed_imgs = []
                for full_img in full_res_imgs:
                    # Step 2: Downscale the full-resolution image to a lower resolution.
                    low_img = full_img.resize(
                        (round(full_img.width / hiresfix_scale), round(full_img.height / hiresfix_scale)),
                        resample=Image.LANCZOS
                    )

                    # Step 3: Upscale the downscaled image back using RealESRGAN.
                    low_img_np = np.array(low_img)[..., ::-1]  # Convert to BGR
                    sr_img_np, _ = upsampler.enhance(low_img_np, outscale=int(hiresfix_scale))
                    sr_img = Image.fromarray(sr_img_np[..., ::-1])  # Back to RGB

                    refine_kwargs = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "image": sr_img,  # for img2img, use "image"
                        "strength": strength,
                        "num_inference_steps": steps,
                        "guidance_scale": cfg_scale,
                        "guidance_rescale": guidance_rescale,
                        "pag_scale": pag_scale,
                        "clip_skip": clip_skip - 1,
                        "generator": torch.Generator(device="cuda").manual_seed(seed),
                    }

                    refine_pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
                    refined_img = refine_pipeline(**refine_kwargs)["images"][0]

                    fixed_imgs.append(refined_img)
                imgs = fixed_imgs

            # --- Apply GFPGAN if enabled (Runs after all generation types) ---
            if face_restoration and gfpganer:
                print("Applying GFPGAN face restoration to final images...")
                restored_imgs = []
                for img in imgs:
                    try:
                        _, _, restored_img_np = gfpganer.enhance(
                            np.array(img.convert('RGB')), # Ensure RGB
                            has_aligned=False,
                            only_center_face=False,
                            paste_back=True
                        )
                        restored_img = Image.fromarray(restored_img_np)
                        restored_imgs.append(restored_img)
                    except Exception as e:
                        print(f"Warning: GFPGAN enhance failed for an image. Skipping. Error: {e}")
                        restored_imgs.append(img) # Keep original if enhance fails
                imgs = restored_imgs # Replace original list with restored list
            # --- End GFPGAN Application ---

            image_paths = []
            # Import the metadata writing utility.
            for index, img in enumerate(imgs):
                img_file_path = f"tmp/{index}.png"
                # Save the image using the specified compression level and optimization flag.
                img.save(img_file_path, optimize=optimize, compress_level=compress_level)
                image_paths.append(Path(img_file_path))
                
                # Capture generation metadata details.
                metadata = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model": model,
                    "vae": vae if vae else BAKEDIN_VAE_LABEL,  # Adjust as required.
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "guidance_rescale": guidance_rescale,
                    "pag_scale": pag_scale,
                    "clip_skip": clip_skip,
                    "seed": seed,
                    "scheduler": scheduler,
                    "hiresfix": hiresfix,
                    "lora": loras,
                    "face_restoration": face_restoration,
                    "gfpgan_model": gfpgan_model,
                    "hiresfix_scale": hiresfix_scale,
                    "strength": strength,
                    "width": width,
                    "height": height,
                    "hiresfix_model": hiresfix_model,
                    "blur_factor": blur_factor,
                    "prepend_preprompt": prepend_preprompt,
                    "prompt_emebding": prompt_emebding,
                    # Add refiner status to metadata
                    "refiner_enabled": should_use_refiner, 
                    "refiner_high_noise_frac": high_noise_frac if should_use_refiner else None,
                    # Add additional parameters you'd like to track.
                }
                utils.write_generation_metadata(Path(img_file_path), metadata)
            return image_paths
        finally:
            pipeline.unload_lora_weights()

class SDXLMultiPipelineHandler:

    def __init__(self, model_name_obj_dict, vaes_dir_path, vae_names, textual_inversion_paths, torch_dtype, cpu_offload_inactive_models):
        print("--- SDXLMultiPipelineHandler: Initializing ---") # Debug
        self.model_name_obj_dict = model_name_obj_dict
        self.model_pipeline_dict = {} # Key = Model's name(str), Value = StableDiffusionXLPipeline instance.
        self.vaes_dir_path = vaes_dir_path
        self.vae_obj_dict = {vae_name: None for vae_name in vae_names} # Key = VAE's name(str), Value = AutoencoderKL instance.
        self.textual_inversion_paths = textual_inversion_paths
        self.torch_dtype = torch_dtype
        self.cpu_offload_inactive_models = cpu_offload_inactive_models
        self.refiner_pipeline = None # Placeholder for refiner

        print("--- SDXLMultiPipelineHandler: Starting VAE loading ---") # Debug
        self._load_all_vaes() # Must load VAEs before models.
        print("--- SDXLMultiPipelineHandler: Finished VAE loading ---") # Debug
        
        print("--- SDXLMultiPipelineHandler: Starting Model loading ---") # Debug
        self._load_all_models()
        print("--- SDXLMultiPipelineHandler: Finished Model loading ---") # Debug
        
        print("--- SDXLMultiPipelineHandler: Starting Refiner loading ---") # Debug
        self._load_refiner() # Load the refiner model
        print("--- SDXLMultiPipelineHandler: Finished Refiner loading ---") # Debug

        self.activated_model = None
        print("--- SDXLMultiPipelineHandler: Initialization Complete ---") # Debug

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
        # Explicitly enable xformers memory efficient attention
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print(f"xformers memory efficient attention enabled for {model_name}.")
        except Exception as e:
            print(f"Warning: Could not enable xformers memory efficient attention for {model_name}. Is xformers installed? Error: {e}")
        
        pipeline.vae = vae
        pipeline.scheduler = SDXLCompatibleSchedulers.create_instance(scheduler_name)
        
        # Also ensure refiner is on cuda if base model is activated
        if self.refiner_pipeline:
             self.refiner_pipeline.to("cuda")
             
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
        self.vae_obj_dict[model_name] = vae
        return pipeline

    def _load_refiner(self):
        """Loads the SDXL Refiner model."""
        # --- Add check for default model pipeline existence ---
        if DEFAULT_MODEL not in self.model_pipeline_dict or not self.model_pipeline_dict[DEFAULT_MODEL]:
            print(f"Warning: Default model ({DEFAULT_MODEL}) pipeline not found or failed to load. Cannot load Refiner.")
            self.refiner_pipeline = None
            return
        # --- End check ---
        
        try:
            print("Loading SDXL Refiner model...")
            # Reuse components from the *loaded* default model pipeline
            refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=self.model_pipeline_dict[DEFAULT_MODEL].text_encoder_2, # Reuse text encoder 2
                vae=self.vae_obj_dict[DEFAULT_MODEL], # Get VAE object stored during base model loading
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.torch_dtype == torch.float16 else None # Use fp16 variant if appropriate
            )
            
            if not self.cpu_offload_inactive_models:
                refiner.to("cuda")
            self.refiner_pipeline = refiner
            print("SDXL Refiner model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load SDXL Refiner model. Refiner will be unavailable. Error: {e}")
            self.refiner_pipeline = None

# For RealESRGAN_x4plus_anime_6B, create the backbone architecture
def get_upscale_backbone(model_name: str):
    # Create the backbone architecture based on the model name.
    if model_name == "RealESRGAN_x4plus_anime_6B":
        # For the anime model, use a smaller RRDBNet (fewer blocks)
        backbone = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,       # Fewer blocks for the anime version
            num_grow_ch=32,
            scale=4
        )
        netscale = 4
    else:
        raise NotImplementedError("Model name not supported")
    return backbone, netscale
