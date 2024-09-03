import gradio as gr

from core import download_model
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def download_model_tab():
    with gr.Row():
        link = gr.Textbox(
            label="Model URL",
            lines=1,
            type="url",
        )
    output = gr.Textbox(
        label=i18n("Output Information"),
        info=i18n("The output information will be displayed here."),
    )
    download = gr.Button(i18n("Download"))

    download.click(
        download_model,
        inputs=[link],
        outputs=[output],
    )
