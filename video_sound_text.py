import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
from moviepy import VideoFileClip
from glob import glob
import json
import os

torch_dtype = torch.float16
device = "cuda"

# Load the model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")

# Set up the pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device=device,
    return_timestamps=True
)

# Path to your video files
video_path = glob("Videos/*")
tr_data = {}

for f in video_path:
    # Load the video clip
    video_clip = VideoFileClip(f)

    # Extract audio from the video clip
    audio_clip = video_clip.audio

    # Save the audio as a WAV file
    audio_clip.write_audiofile('tmp_audio.wav')

    # Load your local WAV file using torchaudio
    audio_path = 'tmp_audio.wav'  # Replace this with the path to your local .wav file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:  # Check if audio is stereo (2 channels)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono by averaging channels

    # Convert the waveform to the correct format for the pipeline
    sample = {
        "array": waveform.squeeze().cpu().numpy(),
        "sampling_rate": sample_rate
    }

    # Run the pipeline
    result = pipe(sample)

    tr_data[f] = result["text"]

# saving video audio transcript
# Ensure the directory exists
os.makedirs("Outputs", exist_ok=True)
with open("Outputs/transcript_data.json", "w", encoding="utf-8") as json_file:
    json.dump(tr_data, json_file, indent=4)

# delete tmp audio file
os.remove("tmp_audio.wav")