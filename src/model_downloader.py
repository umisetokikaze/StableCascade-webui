import sys
import requests
import os
import requests
from tqdm import tqdm



def download_file(url, path="./models"):
    local_filename = url.split('/')[-1]
    full_path = f"{path}/{local_filename}"

    # ファイルが既に存在するかチェック
    if os.path.exists(full_path):
        return

    # レスポンスヘッダからファイルサイズを取得
    response = requests.head(url)
    total_size_in_bytes = int(response.headers.get('content-length', 0))

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True,
            desc=f"{local_filename}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as progress:
            with open(full_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    progress.update(len(chunk))
                    f.write(chunk)
    print(f"Downloaded {local_filename} to {path}")

def download_model(essential, model_size, presion):
        # Check for the optional "essential" argument and download the essential models if present
    if essential == True:
        print("Downloading Essential Models (EfficientNet, Stage A, Previewer)")
        download_file("https://huggingface.co/stabilityai/StableWurst/resolve/main/stage_a.safetensors")
        download_file("https://huggingface.co/stabilityai/StableWurst/resolve/main/previewer.safetensors")
        download_file("https://huggingface.co/stabilityai/StableWurst/resolve/main/effnet_encoder.safetensors")

    base_url = "https://huggingface.co/stabilityai/StableWurst/resolve/main/"
    model_sizes = {
        "big-big": [("stage_b_bf16.safetensors", "stage_c_bf16.safetensors") if presion == "bf16" else ("stage_b.safetensors", "stage_c.safetensors")],
        "big-small": [("stage_b_bf16.safetensors", "stage_c_lite_bf16.safetensors") if presion == "bf16" else ("stage_b.safetensors", "stage_c_lite.safetensors")],
        "small-big": [("stage_b_lite_bf16.safetensors", "stage_c_bf16.safetensors") if presion == "bf16" else ("stage_b_lite.safetensors", "stage_c.safetensors")],
        "small-small": [("stage_b_lite_bf16.safetensors", "stage_c_lite_bf16.safetensors") if presion == "bf16" else ("stage_b_lite.safetensors", "stage_c_lite.safetensors")]
    }

    if model_size in model_sizes:
        for filename in model_sizes[model_size][0]:
            download_file(base_url + filename)
    else:
        print("Invalid second argument. Please provide a valid argument: big-big, big-small, small-big, or small-small.")
        sys.exit(2)

