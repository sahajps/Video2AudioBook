import numpy as np
import re
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf
from tqdm import tqdm
import os
import json

device = "cuda:6" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")

def diff_text_to_sets(text):
    # Define the regex pattern to match gendered dialogues
    dialog_pattern = r"\(([\w\s]+):\s([^)]+)\)"
    
    # Split the text into lines
    lines = text.split('\n')
    
    # Initialize an empty list to hold sets of labels and content
    diff_sets = []
    
    # Iterate through each line
    for line in lines:
        line = line.strip()
        
        # Search for a match in the line using the defined regex pattern
        match = re.match(dialog_pattern, line)
        
        if match:
            # Extract the label (gender word) and the dialog content
            label = match.group(1)
            content = match.group(2)
            # Add the tuple to the list of sets
            diff_sets.append((label, content))
        elif line:  # For direct narration (non-dialogue lines)
            # Treat it as narrator content
            diff_sets.append(("narrator", line))
    
    return diff_sets

#####################################################################
with open("Outputs/audiobook_text_data.json", "r", encoding="utf-8") as json_file:
    ab_data = json.load(json_file)

# Create 1 second of silence (0 values) to add delay
silence = np.zeros(int(model.config.sampling_rate))  # 1 second of silence

for p in ab_data.keys():
    # To store the final concatenated audio array
    final_audio = np.array([])
    dialogs = diff_text_to_sets(ab_data[p])
    for i in tqdm(range(len(dialogs))):
        prompt = dialogs[i][1]
        if dialogs[i][0]=="narrator":
            description = "A man with heavy voice who is a narrator of an audiobook."
        else:
            description = f"A {dialogs[i][0]} voice who is speaking their dilogs as part of an audiobook."

        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        set_seed(42)
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        # Concatenate the audio with the silence (if it's not the first audio, add silence before)
        if i == 0:
            final_audio = audio_arr  # First audio, no silence before
        else:
            final_audio = np.concatenate([final_audio, silence, audio_arr])

        # sf.write(f"Outputs/Logs/{i}.wav", audio_arr, model.config.sampling_rate)
    # Save the final concatenated audio with delays
    os.makedirs("Outputs/Audiobooks", exist_ok=True)
    ab_name = p.split("/")[-1].split(".")[0]
    sf.write(f"Outputs/Audiobooks/{ab_name}.wav", final_audio, model.config.sampling_rate)