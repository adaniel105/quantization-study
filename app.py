import gradio as gr
import whisper 
import os
import shutil
import subprocess

def SRTGen(video):
    model = whisper.load_model("tiny")
    result = model.transcribe(video)
    srt_file_path = str("/tmp/gradio/")
    writer = whisper.utils.get_writer("srt", srt_file_path)
    writer(result, video)
    video_name = os.path.basename(video)
    srt_file = f"{srt_file_path}/{video_name}.srt"
    return result["text"], srt_file

#SRTGen("/content/science.mp4")

demo = gr.Interface(fn=SRTGen, inputs=[gr.Video(label="Upload Video")], outputs=["text", "file"])
demo.launch(debug=True)