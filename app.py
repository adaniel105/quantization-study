import gradio as gr
import whisper
import os
import shutil
import subprocess

'''
def SRTGen(video, src_lang, tgt_lang):
    model = whisper.load_model("tiny")
    result = model.transcribe(video)
    srt_file_path = str("/tmp/gradio/")
    writer = whisper.utils.get_writer("srt", srt_file_path)
    writer(result, video)
    video_name = os.path.basename(video).split(".")[0]
    srt_file = f"{srt_file_path}/{video_name}.srt"
    return result["text"], srt_file
'''

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Upload Video")
            src_lang = gr.Text(label="Source Language")
            tgt_lang = gr.Dropdown([("English", "en") , ("Hausa", "ha") , ("Yoruba", "yo")], label="Target Language")
            text_submit = gr.Button("Generate Text")
        with gr.Column():
            gr.Text(label="Transcription", placeholder="Text output here.....")
            srt_submit = gr.Button("Generate SRT")
            with gr.Column():
                file_output = gr.File(label="SRT File")

    text_submit.click(fn=SRTGen, inputs=[video, src_lang, tgt_lang], outputs=[gr.Text(label="Transcription")])
    srt_submit.click(fn=srt_dl, inputs=[file_output], outputs=[file_output])


#demo = gr.Interface(fn=SRTGen, inputs=[gr.Video(label="Upload Video")], outputs=["text", "file"])
demo.launch(debug=True)
