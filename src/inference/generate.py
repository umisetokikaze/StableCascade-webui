
import os
import random
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

def load_config(config_name, config_path='./configs/inference'):
    config_file = os.path.join(config_path, f'{config_name}.yaml')
    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def initialize_model(core_class, config, device, training=False,Bmode=False):
    core = core_class(config_dict=config, device=device, training=training)
    extras = core.setup_extras_pre()
    models = core.setup_models(extras)
    models.generator.eval().requires_grad_(False)
    
    if Bmode:
        models = WurstCoreB.Models(
        **{**models.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
        )
        models.generator.bfloat16().eval().requires_grad_(False)
        print("STAGE B READY")
        return core,models,extras
    print("STAGE C READY")
    return core,models,extras

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

def generate(core, core_b, models, models_b, extras, extras_b, caption,batch_size, stage_c_latent_shape, stage_b_latent_shape, device,seed=42, outdir='output'):
    os.makedirs(outdir, exist_ok=True)
    seed = random.randint(0, 999999) if seed == -1 else seed
    # PREPARE CONDITIONS
    batch = {'captions': [caption] * batch_size}
    conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
    unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        torch.manual_seed(seed)
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

def t2i(batch_size, caption, height, width, presion,model_size,essential,outdir,seed,cfg_c,cfg_b,shift_c,shift_b,step_c,step_b):
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    download_model(essential, model_size, presion)
    c_model_size, b_model_size = determine_model_sizes(model_size)
    config_c,config_b = f'stage_c_{c_model_size}_{presion}',f'stage_b_{b_model_size}_{presion}'
    config_c, config_b = load_config(config_c), load_config(config_b)
    
    core, models, extras = initialize_model(WurstCoreC, config_c, device)
    core_b, models_b, extras_b = initialize_model(WurstCoreB, config_b, device,False,True)
    
    setup_sampling_configs(extras,cfg=cfg_c, shift=shift_c, timesteps=step_c)
    setup_sampling_configs(extras_b, cfg=cfg_b, shift=shift_b, timesteps=step_b)
    stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)
    images = generate(core, core_b, models, models_b, extras, extras_b, caption,batch_size, stage_c_latent_shape, stage_b_latent_shape, device,seed,outdir)
    return images