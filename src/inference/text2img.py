import os
import yaml
import torch
from tqdm import tqdm
import sys
import time

from model_downloader import download_model

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from inference.utils import *
from core import load_or_fail
from train import WurstCoreC, WurstCoreB

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def load_config(c_model_config, b_model_config):
    # SETUP STAGE C
    config_file = f'./configs/inference/{c_model_config}.yaml'
    with open(config_file, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

    core = WurstCoreC(config_dict=loaded_config, device=device, training=False)

    # SETUP STAGE B
    config_file_b = f'./configs/inference/{b_model_config}.yaml'
    with open(config_file_b, "r", encoding="utf-8") as file:
        config_file_b = yaml.safe_load(file)
        
    core_b = WurstCoreB(config_dict=config_file_b, device=device, training=False)
    return core, core_b

def load_models(core, core_b):
    # SETUP MODELS & DATA
    extras = core.setup_extras_pre()
    models = core.setup_models(extras)
    models.generator.eval().requires_grad_(False)
    print("STAGE C READY")

    extras_b = core_b.setup_extras_pre()
    models_b = core_b.setup_models(extras_b, skip_clip=True)
    models_b = WurstCoreB.Models(
    **{**models_b.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
    )
    models_b.generator.bfloat16().eval().requires_grad_(False)
    print("STAGE B READY")
    return models, models_b, extras, extras_b

def determine_model_sizes(model_type):
    if model_type == "big-big":
        c_model_size = "full"
        b_model_size = "full"
    elif model_type == "big-small":
        c_model_size = "lite"
        b_model_size = "full"
    elif model_type == "small-big":
        c_model_size = "full"
        b_model_size = "lite"
    elif model_type == "small-small":
        c_model_size = "lite"
        b_model_size = "lite"
    else:
        # 不正なmodel_typeが指定された場合はエラーメッセージを返す
        return "Error: Invalid model type specified."
    return c_model_size,b_model_size

def generate(batch_size, caption, height, width, presion,model_size,essential):
    download_model(essential, model_size, presion)
    c_model_size, b_model_size = determine_model_sizes(model_size)
    os.makedirs('output', exist_ok=True)
    core, core_b = load_config(f'stage_c_{c_model_size}_{presion}',f'stage_b_{b_model_size}_{presion}')
    models, models_b, extras, extras_b = load_models(core, core_b)
    stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

    # Stage C Parameters
    extras.sampling_configs['cfg'] = 4
    extras.sampling_configs['shift'] = 2
    extras.sampling_configs['timesteps'] = 20
    extras.sampling_configs['t_start'] = 1.0

    # Stage B Parameters
    extras_b.sampling_configs['cfg'] = 1.1
    extras_b.sampling_configs['shift'] = 1
    extras_b.sampling_configs['timesteps'] = 10
    extras_b.sampling_configs['t_start'] = 1.0

    # PREPARE CONDITIONS
    batch = {'captions': [caption] * batch_size}
    conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
    unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # torch.manual_seed(42)
        sampling_c = extras.gdf.sample(
            models.generator, conditions, stage_c_latent_shape,
            unconditions, device=device, **extras.sampling_configs,
        )   
        for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs['timesteps']):
            sampled_c = sampled_c
            
        # preview_c = models.previewer(sampled_c).float()
        # show_images(preview_c)

        conditions_b['effnet'] = sampled_c
        unconditions_b['effnet'] = torch.zeros_like(sampled_c)

        sampling_b = extras_b.gdf.sample(
            models_b.generator, conditions_b, stage_b_latent_shape,
            unconditions_b, device=device, **extras_b.sampling_configs
        )
        for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
            sampled_b = sampled_b
        sampled = models_b.stage_a.decode(sampled_b).float()
        # save_image
        times = time.strftime(r"%Y%m%d%H%M%S")
        imgs = []
        for i, img in enumerate(sampled):
            img = torchvision.transforms.functional.to_pil_image(img.clamp(0, 1))
            fileName = f"img-{times}-{i+1}.png"
            img.save(f"output/{fileName}")
            imgs.append(img)
        return imgs

