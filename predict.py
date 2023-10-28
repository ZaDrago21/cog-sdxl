from cog import BasePredictor, Input, Path
from typing import List
import os
import glob
import torch
import diffusers
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

def find_models(models_dir):
    return [os.path.basename(file) for file in glob.glob(f"{models_dir}/**/*.safetensors", recursive=True) + glob.glob(f"{models_dir}/**/*.ckpt", recursive=True)]

def find_vaes(vaes_dir):
    vae_names = []
    for folder in os.listdir(vaes_dir):
        folder_path = os.path.join(vaes_dir, folder)
        if os.path.isdir(folder_path):
            safetensors_file = os.path.join(folder_path, 'diffusion_pytorch_model.safetensors')
            bin_file = os.path.join(folder_path, 'diffusion_pytorch_model.bin')
            config_file = os.path.join(folder_path, 'config.json')
            if (os.path.isfile(safetensors_file) or os.path.isfile(bin_file)) and os.path.isfile(config_file):
                vae_names.append(folder)
    return vae_names

MODELS_DIR_PATH = "models"
VAES_DIR_PATH = "vaes"
MODEL_NAMES = find_models(MODELS_DIR_PATH)
assert len(MODEL_NAMES) > 0, f"You don't have any model under \"{MODELS_DIR_PATH}\", please put at least 1 model in there."
VAE_NAMES = find_vaes(VAES_DIR_PATH)
assert len(VAE_NAMES) > 0, f"You don't have any VAE under \"{VAES_DIR_PATH}\", please put at least 1 VAE in there, you can run \"python3 -c 'from huggingface_hub import snapshot_download as d;d(repo_id=\"madebyollin/sdxl-vae-fp16-fix\", allow_patterns=[\"config.json\", \"diffusion_pytorch_model.safetensors\"], local_dir=\"./vaes/sdxl-vae-fp16-fix\", local_dir_use_symlinks=False)'\" to download a fp16 fixed default SDXL VAE if you don't know what to use."

# Cog will only run this class in a single thread
class Predictor(BasePredictor):

    def setup(self):
        self.pipelines = SDXLMultiPipelineSwitchAutoDetect(MODELS_DIR_PATH, MODEL_NAMES, VAES_DIR_PATH, VAE_NAMES)
        os.makedirs("tmp", exist_ok=True)

    @torch.no_grad
    def predict(
        self,
        model: str = Input(description="The model to use", default=MODEL_NAMES[0], choices=MODEL_NAMES),
        vae: str = Input(description="The VAE to use", default=VAE_NAMES[0], choices=VAE_NAMES),
        prompt: str = Input(description="The prompt", default="catgirl, cat ears, white hair, golden eyes, bob cut, pov, face closeup, smile"),
        negative_prompt: str = Input(description="The negative prompt (For things you don't want)", default="lowres, low quality, worse quality, monochrome, blurry, headphone, big breasts"),
        steps: int = Input(description="The steps when generating", default=35, ge=1, le=100),
        cfg_scale: float = Input(description="CFG Scale defines how much attention the model pays to the prompt when generating", default=7, ge=1, le=30),
        guidance_rescale: float = Input(description="The amount to rescale CFG generated noise to avoid generating overexposed images", default=0.7, ge=0, le=1),
        width: int = Input(description="The width of the image", default=1184, ge=1, le=2048),
        height: int = Input(description="The height of the image", default=864, ge=1, le=2048),
        batch_size: int = Input(description="Number of images to generate (1-4)", default=1, ge=1, le=4),
        seed: int = Input(description="The seed used when generating, set to -1 for random seed", default=-1),
    ) -> List[Path]:
        pipeline = self.pipelines.get_pipeline(model, vae)
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        imgs = pipeline(
            prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, num_inference_steps=steps,
            guidance_scale=cfg_scale, guidance_rescale=guidance_rescale, num_images_per_prompt=batch_size, generator=generator,
        ).images

        image_paths = []
        for index, img in enumerate(imgs):
            img_file_path = f"tmp/{index}.png"
            img.save(img_file_path, compression=9)
            image_paths.append(Path(img_file_path))

        return image_paths

class SDXLMultiPipelineSwitchAutoDetect:

    def __init__(self, models_dir_path, model_names, vaes_dir_path, vae_names):
        self.models_dir_path = models_dir_path
        self.model_pipeline_dict = {model_name: None for model_name in model_names}
        self.vaes_dir_path = vaes_dir_path
        self.vae_obj_dict = {vae_name: None for vae_name in vae_names}
        self._load_all_models()
        self._load_all_vaes()
        self.on_cuda_model = model_names[0]
        on_cuda_pipeline = self.model_pipeline_dict[self.on_cuda_model]
        on_cuda_pipeline.to("cuda")
        on_cuda_pipeline.vae = self.vae_obj_dict[vae_names[0]]

    def get_pipeline(self, model_name, vae_name):
        pipeline = self.model_pipeline_dict.get(model_name)
        vae = self.vae_obj_dict.get(vae_name)
        # __init__ function guarantees models and VAEs to be loaded.
        if pipeline is None:
            raise ValueError(f"Model '{model_name}' not found.")
        if vae is None:
            raise ValueError(f"VAE '{vae_name}' not found.")

        if model_name != self.on_cuda_model:
            on_cuda_pipeline = self.model_pipeline_dict[self.on_cuda_model]
            on_cuda_pipeline.vae = None
            on_cuda_pipeline.to("cpu")
            pipeline.to("cuda")
            self.on_cuda_model = model_name

        pipeline.vae = vae
        return pipeline

    def _load_all_models(self):
        for model_name in self.model_pipeline_dict.keys():
            pipeline = self._load_model(model_name)
            self.model_pipeline_dict[model_name] = pipeline

    def _load_model(self, model_name):
        pipeline = StableDiffusionXLPipeline.from_single_file(os.path.join(self.models_dir_path, model_name), torch_dtype=torch.bfloat16, variant="fp16", add_watermarker=False)
        pipeline.vae = None
        pipeline.scheduler = diffusers.UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_xformers_memory_efficient_attention()
        return pipeline

    def _load_all_vaes(self):
        for vae_name in self.vae_obj_dict.keys():
            vae = self._load_vae(vae_name)
            self.vae_obj_dict[vae_name] = vae

    def _load_vae(self, vae_name):
        vae = AutoencoderKL.from_pretrained(os.path.join(self.vaes_dir_path, vae_name), torch_dtype=torch.bfloat16)
        vae.enable_slicing()
        vae.enable_tiling()
        vae.to("cuda")
        return vae