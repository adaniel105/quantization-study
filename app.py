import gradio as gr
import whisper
import os
import shutil
import subprocess

model = whisper.load_model("tiny") #replace with large-v3

def srt_gen(video, tgt_lang):
  if tgt_lang == "en":
    result = model.transcribe(video, language=tgt_lang)
  else:
    pass
  return result["text"]

def srt_dl(video, tgt_lang):
    result = model.transcribe(video)
    srt_file_path = str("/tmp/gradio/")
    writer = whisper.utils.get_writer("srt", srt_file_path)
    writer(result, video)
    video_name = os.path.basename(video).split(".")[0]
    srt_file = f"{srt_file_path}/{video_name}.srt"
    return srt_file

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Upload Video")
            tgt_lang = gr.Dropdown([("English", "en") , ("Hausa", "ha") , ("Yoruba", "yo")], label="Target Language")
            text_submit = gr.Button("Generate Text", variant="primary")
        with gr.Column():
            text_output = gr.Text(label="Transcription", placeholder="Text output here.....")
            srt_submit = gr.Button("Generate Subtitles", variant="primary")
            with gr.Column():
                srt_output = gr.File(label="Subtitles")

    text_submit.click(fn=srt_gen, inputs=[video, tgt_lang], outputs=[text_output])
    srt_submit.click(fn=srt_dl, inputs=[video, tgt_lang], outputs=[srt_output])

demo.launch(debug=True)