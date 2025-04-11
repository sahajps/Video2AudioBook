# This code is taken from/inspired by the demo: https://colab.research.google.com/drive/1CZggLHrjxMReG-FNOmqSOdi4z7NPq6SO?usp=sharing

from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch
import av
import numpy as np
from glob import glob
import json
import os

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "/home/sahajps/Models/LLaVA-NeXT-Video-7B-hf",
    torch_dtype=torch.float16,
    device_map='cuda'
)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

###################################################################
files = glob("Videos/*")

desc_data = {}
for f in files:
    container = av.open(f)

    # sample uniformly 8 frames from the video (we can sample more for longer videos)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    context_video = read_video_pyav(container, indices)

    # Each "content" is a list of dicts and you can add image/video/text modalities
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Provide a detailed description of each scene in the video, capturing key visual elements, actions, and interactions. Include information about the setting, characters, objects, and any notable movements or expressions. If applicable, describe the lighting, colors, and atmosphere. Additionally, mention any background activities, transitions between scenes, and significant audio cues or background sounds that contribute to the overall mood of the video. Aim to create a vivid and comprehensive summary that allows someone who hasn’t seen the video to understand its content clearly."},
                {"type": "video"},
                ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # we still need to call the processor to tokenize the prompt and get pixel_values for videos
    inputs = processor([prompt], videos=[context_video], padding=True, return_tensors="pt").to(model.device)

    generate_kwargs = {"max_new_tokens": 2500, "do_sample": True, "top_p": 0.9}

    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)

    desc_data[f] = generated_text[0].split("ASSISTANT: ")[1]

# saving video desc
# Ensure the directory exists
os.makedirs("Outputs", exist_ok=True)
with open("Outputs/description_data.json", "w", encoding="utf-8") as json_file:
    json.dump(desc_data, json_file, indent=4)