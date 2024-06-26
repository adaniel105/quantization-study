import gradio as gr
import whisper
import os
import pysrt
import shutil
import subprocess

model = whisper.load_model("large-v3")


def srt_gen(video, src_lang, tgt_lang):
    if src_lang == "en" and tgt_lang != "en":
        translations = MultiInference(video, tgt_lang)
        return translations
    else:
        result = model.transcribe(video, language=tgt_lang)
        return result["text"]


def srt_dl(video, src_lang, tgt_lang):
    video_name = os.path.basename(video).split(".")[0]
    if src_lang == "en" and tgt_lang != "en":
        subs = pysrt.SubRipFile()
        translations = MultiInference(video, tgt_lang)
        for i in range(len(translations["chunks"])):
            start_time = translations["chunks"][i]["timestamp"][0]
            end_time = translations["chunks"][i]["timestamp"][1]
            text = translations["chunks"][i]["text"]
            start = float_to_subriptime(start_time)
            end = float_to_subriptime(end_time)
            sub_item = pysrt.SubRipItem(
                index=i, start=start, end=end, text=text)
            subs.append(sub_item)
            subs.save(f'/tmp/gradio/{video_name}.srt', encoding='utf-8')
    else:
        result = model.transcribe(video)
        srt_file_path = str("/tmp/gradio/")
        writer = whisper.utils.get_writer("srt", srt_file_path)
        writer(result, video)
        srt_file = f"{srt_file_path}/{video_name}.srt"
        return srt_file


with gr.Blocks() as demo:
    gr.Markdown("# Automated Captions Generation")
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Upload Video")
            src_lang = gr.Dropdown(
                [("English", "en"), ("Hausa", "ha"), ("Yoruba", "yo")], label="Source Language")
            tgt_lang = gr.Dropdown(
                [("English", "en"), ("Hausa", "ha"), ("Yoruba", "yo")], label="Target Language")
            text_submit = gr.Button("Generate Text", variant="primary")
        with gr.Column():
            text_output = gr.Text(label="Transcription",
                                  placeholder="Text output here.....")
            srt_submit = gr.Button("Generate Subtitles", variant="primary")
            with gr.Column():
                srt_output = gr.File(label="Subtitles")

    text_submit.click(fn=srt_gen, inputs=[
                      video, src_lang, tgt_lang], outputs=[text_output])
    srt_submit.click(fn=srt_dl, inputs=[video, tgt_lang], outputs=[srt_output])

demo.launch(debug=True)
