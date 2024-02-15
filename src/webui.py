import gradio as gr

from inference.text2img import generate
def webui():
    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                caption = gr.TextArea(label="Caption")
                batch_size = gr.Slider(1, 10, 4,step=1, label="Batch Size")
                height = gr.Slider(64, 2048, 1024,step=2, label="Height")
                width = gr.Slider(64, 2048, 1024,step=2, label="Width")
            with gr.Column():
                output =gr.Gallery(caption="Output Image")
                run = gr.Button(text="Run")
            run.click(fn=generate, inputs=[batch_size, caption, height, width], outputs=[output])
            ui.launch()

webui()