import PIL
import torch
import requests
import torchvision
from math import ceil
from io import BytesIO
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def download_image(url):
    return PIL.Image.open(requests.get(url, stream=True).raw).convert("RGB")


def resize_image(image, size=768):
    tensor_image = F.to_tensor(image)
    resized_image = F.resize(tensor_image, size, antialias=True)
    return resized_image


def downscale_images(images, factor=3/4):
    scaled_height, scaled_width = int(((images.size(-2)*factor)//32)*32), int(((images.size(-1)*factor)//32)*32)
    scaled_image = torchvision.transforms.functional.resize(images, (scaled_height, scaled_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    return scaled_image


def calculate_latent_sizes(height=1024, width=1024, batch_size=4, compression_factor_b=42.67, compression_factor_a=4.0):
    resolution_multiple = 42.67
    latent_height = ceil(height / compression_factor_b)
    latent_width = ceil(width / compression_factor_b)
    stage_c_latent_shape = (batch_size, 16, latent_height, latent_width)
    
    latent_height = ceil(height / compression_factor_a)
    latent_width = ceil(width / compression_factor_a)
    stage_b_latent_shape = (batch_size, 4, latent_height, latent_width)
    
    return stage_c_latent_shape, stage_b_latent_shape
