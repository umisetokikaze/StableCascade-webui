import random
import gradio as gr
from inference.generate import t2i
import argparse  # argparseをインポート

# コマンドライン引数を解析するための関数を定義
def parse_args():
    parser = argparse.ArgumentParser(description="Launch Gradio UI for text-to-image generation.")
    parser.add_argument("--share", action="store_true", help="If set, share the Gradio app publicly.")
    parser.add_argument("--listen", action="store_true", help="If set, listen on all network interfaces (0.0.0.0).")
    return parser.parse_args()

def webui(share=False, listen=False): 
    with gr.Blocks() as ui:
        with gr.Row():
            precision = gr.Dropdown(value="bf16", choices=["bf16", "fp32"], label="Precision")
            model_size = gr.Dropdown(value="big-small", choices=["big-big", "big-small", "small-big", "small-small"], label="Model size")
            essential = gr.Checkbox(label="Download essential models", value=True)
        with gr.Row():
            with gr.Column():
                caption = gr.TextArea(label="Caption")
                batch_size = gr.Slider(1, 10, 4, step=1, label="Batch Size")
                height = gr.Slider(64, 2048, 1024, step=2, label="Height")
                width = gr.Slider(64, 2048, 1024, step=2, label="Width")
                seed = gr.Number(-1, 9999, -1, step=1, label="Seed")
                cfg_c = gr.Slider(1, 20, 4, step=0.1, label="cfg_c")
                cfg_b = gr.Slider(1, 20, 1.1, step=0.1, label="cfg_b")
                shift_c = gr.Slider(1, 7, 1, step=1, label="shift_c")
                shift_b = gr.Slider(1, 7, 2, step=1, label="shift_b")
                step_c = gr.Slider(1, 200, 20, step=2, label="step_c")
                step_b = gr.Slider(1, 200, 10, step=2, label="step_b")
                
                outdir = gr.Textbox(label="Output Directory", value="output")
            with gr.Column():
                output = gr.Gallery(label="Output Image")
                run = gr.Button(value="Run")
            
            run.click(fn=t2i, inputs=[batch_size, caption, height, width, precision, model_size, essential, outdir, seed, cfg_c, cfg_b, shift_c, shift_b, step_c, step_b], outputs=[output])
        
        
        ui.launch(share=share, server_name="0.0.0.0" if listen else None)

if __name__ == "__main__":
    args = parse_args()  # コマンドライン引数を解析
    webui(share=args.share, listen=args.listen)  # shareとlisten引数をwebui関数に渡す