
import os
import sys
import time
import torch
import yaml
from tqdm import tqdm
import torchvision

# Assuming model_downloader, inference.utils, core, and train modules are correctly set up
from model_downloader import download_model
from inference.utils import *
from core import load_or_fail
from train import WurstCoreC, WurstCoreB

def setup_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_yaml_config(config_name, config_path='./configs/inference'):
    config_file = os.path.join(config_path, f'{config_name}.yaml')
    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def initialize_model(core_class, config, device, training=False):
    core = core_class(config_dict=config, device=device, training=training)
    extras = core.setup_extras_pre()
    models = core.setup_models(extras)
    models.generator.eval().requires_grad_(False)
    return core, models, extras

def determine_model_sizes(model_type):
    sizes = {
        "big-big": ("full", "full"),
        "big-small": ("lite", "full"),
        "small-big": ("full", "lite"),
        "small-small": ("lite", "lite")
    }
    return sizes.get(model_type, ("Error: Invalid model type specified.",))

def setup_sampling_configs(extras, cfg=4, shift=2, timesteps=20, t_start=1.0):
    extras.sampling_configs.update({
        'cfg': cfg,
        'shift': shift,
        'timesteps': timesteps,
        't_start': t_start
    })

def generate_images(core, core_b, models, models_b, extras, extras_b, batch, stage_c_latent_shape, stage_b_latent_shape, device):
    conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
    unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        sampled_c = core.diffuse_and_sample(models, extras, conditions, unconditions, stage_c_latent_shape, device)
        sampled_b = core_b.diffuse_and_sample(models_b, extras_b, conditions_b, unconditions_b, stage_b_latent_shape, device, effnet=sampled_c)
    return sampled_b

def save_images(images, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    times = time.strftime(r"%Y%m%d%H%M%S")
    file_names = []
    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"img-{times}-{i+1}.png")
        torchvision.transforms.functional.to_pil_image(img.clamp(0, 1)).save(img_path)
        file_names.append(img_path)
    return file_names

# Example usage:
# device = setup_device()
# config_c, config_b = load_yaml_config('your_config_c'), load_yaml_config('your_config_b')
# core, models, extras = initialize_model(WurstCoreC, config_c, device)
# core_b, models_b, extras_b = initialize_model(WurstCoreB, config_b, device)
# model_sizes = determine_model_sizes("big-big")
# setup_sampling_configs(extras)
# setup_sampling_configs(extras_b, cfg=1.1, shift=1, timesteps=10)
# batch = {'captions': ["your caption here"] * batch_size}
# images = generate_images(core, core_b, models, models_b, extras, extras_b, batch, ...)
# save_images(images)