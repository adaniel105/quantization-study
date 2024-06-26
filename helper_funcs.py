import gradio as gr
import os
import sys
import subprocess
import pysrt
import whisper
import tempfile
from transformers import pipeline
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer, WhisperTokenizer, WhisperProcessor, pipeline


def vidrip(video_file, output_ext="mp3"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, "-ar", "16000",  f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)

    return f"{filename}.{output_ext}"

def float_to_subriptime(float_time):
  hours = int(float_time)
  minutes = int((float_time - hours) * 60)
  seconds = int(((float_time - hours) * 60 - minutes) * 60)
  return pysrt.SubRipTime(hours=hours, minutes=minutes, seconds=seconds)



def MultiInference(video,tgt_lang):
  peft_model_id = "laksf/laksf-merge-lora-final"
  peft_config = PeftConfig.from_pretrained("laksf/whisper-large-v3-LORA")
  model = WhisperForConditionalGeneration.from_pretrained(
    peft_model_id, load_in_8bit=True, device_map="auto"
  )

  task = "transcribe"
  tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, task=task)
  processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, task=task)
  feature_extractor = processor.feature_extractor
  pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)
  forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language= tgt_lang, task="transcribe")
  audio_file = vidrip(video)
  translations = pipe(audio_file, return_timestamps= True, chunk_length_s=20, stride_length_s=(4,2),generate_kwargs={"forced_decoder_ids": forced_decoder_ids})

  subs = pysrt.SubRipFile()

  for i in range(len(translations["chunks"])):
      start_time = translations["chunks"][i]["timestamp"][0]
      end_time = translations["chunks"][i]["timestamp"][1]
      text = translations["chunks"][i]["text"]
      start = float_to_subriptime(start_time)
      end = float_to_subriptime(end_time)
      sub_item = pysrt.SubRipItem(index=i, start=start, end=end, text= text)
      subs.append(sub_item)
      subs.save('/content/output.srt', encoding='utf-8')
