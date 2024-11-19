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


######## ATTENTION IS ALL YOU NEED
def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \\( - literal character '('
      \\[ - literal character '['
      \\) - literal character ')'
      \\] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\\(literal\\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """
    import re

    re_attention = re.compile(
        r"""
            \\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|
            \)|]|[^\\()\[\]:]+|:
        """,
        re.X,
    )

    re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res
    
def group_tokens_and_weights(token_ids: list, weights: list, pad_last_block=False):
    """
    Produce tokens and weights in groups and pad the missing tokens

    Args:
        token_ids (list)
            The token ids from tokenizer
        weights (list)
            The weights list from function get_prompts_tokens_with_weights
        pad_last_block (bool)
            Control if fill the last token list to 75 tokens with eos
    Returns:
        new_token_ids (2d list)
        new_weights (2d list)

    Example:
        token_groups,weight_groups = group_tokens_and_weights(
            token_ids = token_id_list
            , weights = token_weight_list
        )
    """
    bos, eos = 49406, 49407

    # this will be a 2d list
    new_token_ids = []
    new_weights = []
    while len(token_ids) >= 75:
        # get the first 75 tokens
        head_75_tokens = [token_ids.pop(0) for _ in range(75)]
        head_75_weights = [weights.pop(0) for _ in range(75)]

        # extract token ids and weights
        temp_77_token_ids = [bos] + head_75_tokens + [eos]
        temp_77_weights = [1.0] + head_75_weights + [1.0]

        # add 77 token and weights chunk to the holder list
        new_token_ids.append(temp_77_token_ids)
        new_weights.append(temp_77_weights)

    # padding the left
    if len(token_ids) > 0:
        padding_len = 75 - len(token_ids) if pad_last_block else 0

        temp_77_token_ids = [bos] + token_ids + [eos] * padding_len + [eos]
        new_token_ids.append(temp_77_token_ids)

        temp_77_weights = [1.0] + weights + [1.0] * padding_len + [1.0]
        new_weights.append(temp_77_weights)

    return new_token_ids, new_weights


def get_prompts_tokens_with_weights(clip_tokenizer: CLIPTokenizer, prompt: str):
    """
    Get prompt token ids and weights, this function works for both prompt and negative prompt

    Args:
        pipe (CLIPTokenizer)
            A CLIPTokenizer
        prompt (str)
            A prompt string with weights

    Returns:
        text_tokens (list)
            A list contains token ids
        text_weight (list)
            A list contains the correspondent weight of token ids

    Example:
        import torch
        from transformers import CLIPTokenizer

        clip_tokenizer = CLIPTokenizer.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , subfolder = "tokenizer"
            , dtype = torch.float16
        )

        token_id_list, token_weight_list = get_prompts_tokens_with_weights(
            clip_tokenizer = clip_tokenizer
            ,prompt = "a (red:1.5) cat"*70
        )
    """
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens, text_weights = [], []
    for word, weight in texts_and_weights:
        # tokenize and discard the starting and the ending token
        token = clip_tokenizer(word, truncation=False).input_ids[1:-1]  # so that tokenize whatever length prompt
        # the returned token is a 1d list: [320, 1125, 539, 320]

        # merge the new tokens to the all tokens holder: text_tokens
        text_tokens = [*text_tokens, *token]

        # each token chunk will come with one weight, like ['red cat', 2.0]
        # need to expand weight for each token.
        chunk_weights = [weight] * len(token)

        # append the weight back to the weight holder: text_weights
        text_weights = [*text_weights, *chunk_weights]
    return text_tokens, text_weights
    
    
def get_weighted_text_embeddings_sdxl(
    pipe: StableDiffusionXLPipeline,
    prompt: str = "",
    prompt_2: str = None,
    neg_prompt: str = "",
    neg_prompt_2: str = None,
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    clip_skip: Optional[int] = None,
    lora_scale: Optional[int] = None,
):
    """
    This function can process long prompt with weights, no length limitation
    for Stable Diffusion XL

    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        prompt_2 (str)
        neg_prompt (str)
        neg_prompt_2 (str)
        num_images_per_prompt (int)
        device (torch.device)
        clip_skip (int)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    """
    device = device or pipe._execution_device

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None and isinstance(pipe, StableDiffusionXLLoraLoaderMixin):
        pipe._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if pipe.text_encoder is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(pipe.text_encoder, lora_scale)
            else:
                scale_lora_layers(pipe.text_encoder, lora_scale)

        if pipe.text_encoder_2 is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(pipe.text_encoder_2, lora_scale)
            else:
                scale_lora_layers(pipe.text_encoder_2, lora_scale)

    if prompt_2:
        prompt = f"{prompt} {prompt_2}"

    if neg_prompt_2:
        neg_prompt = f"{neg_prompt} {neg_prompt_2}"

    prompt_t1 = prompt_t2 = prompt
    neg_prompt_t1 = neg_prompt_t2 = neg_prompt

    if isinstance(pipe, TextualInversionLoaderMixin):
        prompt_t1 = pipe.maybe_convert_prompt(prompt_t1, pipe.tokenizer)
        neg_prompt_t1 = pipe.maybe_convert_prompt(neg_prompt_t1, pipe.tokenizer)
        prompt_t2 = pipe.maybe_convert_prompt(prompt_t2, pipe.tokenizer_2)
        neg_prompt_t2 = pipe.maybe_convert_prompt(neg_prompt_t2, pipe.tokenizer_2)

    eos = pipe.tokenizer.eos_token_id

    # tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(pipe.tokenizer, prompt_t1)
    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt_t1)

    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(pipe.tokenizer_2, prompt_t2)
    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(pipe.tokenizer_2, neg_prompt_t2)

    # padding the shorter one for prompt set 1
    prompt_token_len = len(prompt_tokens)
    neg_prompt_token_len = len(neg_prompt_tokens)

    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens = neg_prompt_tokens + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        neg_prompt_weights = neg_prompt_weights + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
    else:
        # padding the prompt
        prompt_tokens = prompt_tokens + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        prompt_weights = prompt_weights + [1.0] * abs(prompt_token_len - neg_prompt_token_len)

    # padding the shorter one for token set 2
    prompt_token_len_2 = len(prompt_tokens_2)
    neg_prompt_token_len_2 = len(neg_prompt_tokens_2)

    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2 = neg_prompt_tokens_2 + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        neg_prompt_weights_2 = neg_prompt_weights_2 + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
    else:
        # padding the prompt
        prompt_tokens_2 = prompt_tokens_2 + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        prompt_weights_2 = prompt_weights + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)

    embeds = []
    neg_embeds = []

    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(prompt_tokens.copy(), prompt_weights.copy())

    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy(), neg_prompt_weights.copy()
    )

    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy(), prompt_weights_2.copy()
    )

    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy(), neg_prompt_weights_2.copy()
    )

    # get prompt embeddings one by one is not working.
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor([prompt_token_groups[i]], dtype=torch.long, device=device)
        weight_tensor = torch.tensor(prompt_weight_groups[i], dtype=torch.float16, device=device)

        token_tensor_2 = torch.tensor([prompt_token_groups_2[i]], dtype=torch.long, device=device)

        # use first text encoder
        prompt_embeds_1 = pipe.text_encoder(token_tensor.to(device), output_hidden_states=True)

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(token_tensor_2.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds_2[0]

        if clip_skip is None:
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]
            prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-(clip_skip + 2)]
            prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-(clip_skip + 2)]

        prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)

        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                token_embedding[j] = (
                    token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight_tensor[j]
                )

        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding)

        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor([neg_prompt_token_groups[i]], dtype=torch.long, device=device)
        neg_token_tensor_2 = torch.tensor([neg_prompt_token_groups_2[i]], dtype=torch.long, device=device)
        neg_weight_tensor = torch.tensor(neg_prompt_weight_groups[i], dtype=torch.float16, device=device)

        # use first text encoder
        neg_prompt_embeds_1 = pipe.text_encoder(neg_token_tensor.to(device), output_hidden_states=True)
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2]

        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(neg_token_tensor_2.to(device), output_hidden_states=True)
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0)

        for z in range(len(neg_weight_tensor)):
            if neg_weight_tensor[z] != 1.0:
                neg_token_embedding[z] = (
                    neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight_tensor[z]
                )

        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding)

    prompt_embeds = torch.cat(embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    seq_len = negative_prompt_embeds.shape[1]
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(
        bs_embed * num_images_per_prompt, -1
    )
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(
        bs_embed * num_images_per_prompt, -1
    )

    if pipe.text_encoder is not None:
        if isinstance(pipe, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(pipe.text_encoder, lora_scale)

    if pipe.text_encoder_2 is not None:
        if isinstance(pipe, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(pipe.text_encoder_2, lora_scale)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds















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
        
    ) -> list[Path]:
        if prompt == "__ignore__":
            return []
        if prepend_preprompt:
            prompt = DEFAULT_POS_PREPROMPT + prompt
            negative_prompt = DEFAULT_NEG_PREPROMPT + negative_prompt
            
            
        
            
            
        gen_kwargs = {
            "guidance_scale": cfg_scale, "guidance_rescale": guidance_rescale,
            "pag_scale": pag_scale, "clip_skip": clip_skip - 1, "num_inference_steps": steps, "num_images_per_prompt": batch_size,
        }
        pipeline = self.pipelines.get_pipeline(model, None if vae == BAKEDIN_VAE_LABEL else vae, scheduler)
        
        
        try:
            self.loras.process(loras, pipeline)
            
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = get_weighted_text_embeddings_sdxl(
                pipe=pipeline,
                prompt=prompt,
                neg_prompt=negative_prompt,
                num_images_per_prompt=batch_size,
                clip_skip=clip_skip,
                lora_scale=lora_scale,
            )
            
            gen_kwargs["prompt_embeds"] = prompt_embeds
            gen_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
            gen_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
            gen_kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds     
                   
            
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
