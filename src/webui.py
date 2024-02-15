import gradio as gr

from inference.text2img import generate
def webui():
    with gr.Blocks() as ui:
        with gr.Row():
            precision = gr.Dropdown(value="bf16",choices=["bf16","fp32"],label="precision")
            model_size = gr.Dropdown(value="big-small",choices=["big-big","big-small","small-big","small-small"],label="size")
            essential = gr.Checkbox(label="Download essential models")
        with gr.Row():
            with gr.Column():
                caption = gr.TextArea(label="Caption")
                batch_size = gr.Slider(1, 10, 4,step=1, label="Batch Size")
                height = gr.Slider(64, 2048, 1024,step=2, label="Height")
                width = gr.Slider(64, 2048, 1024,step=2, label="Width")
            with gr.Column():
                output =gr.Gallery(label="Output Image")
                run = gr.Button(value="Run")
            run.click(fn=generate, inputs=[batch_size, caption, height, width, precision, model_size,essential], outputs=[output])
            ui.launch()

webui()