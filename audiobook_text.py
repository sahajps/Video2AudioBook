import json
from openai import OpenAI
from timeout_decorator import timeout
import time
import os

try:
    openai_client = OpenAI(api_key="")
except:
    raise ValueError("Please set your OpenAI API key in the code.")

def OpenAI_model_response(prompt, model_name):
    @timeout(15)
    def caller():
        response = openai_client.chat.completions.create(
        model = model_name,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.6
        )
        return response.choices[0].message.content.strip()
    
    while(True):
        try:
            return caller()
        except:
            print("Sleep for 60 sec.")
            time.sleep(60)

######################################################
# Load JSON file into a dictionary
with open("Outputs/description_data.json", "r", encoding="utf-8") as json_file:
    desc_data = json.load(json_file)

with open("Outputs/transcript_data.json", "r", encoding="utf-8") as json_file:
    tr_data = json.load(json_file)

ab_data = {}
for p in desc_data.keys():
    prompt = f"""You are an audiobook creator. Your task is to generate an immersive audiobook adaptation of a given video. The audiobook should be engaging, natural, and well-paced.

**Inputs:**
- **Video Description:** "{desc_data[p]}"
- **Audio Transcript:** "{tr_data[p]}"

**Output Requirements:**
- Convert the spoken text into engaging audiobook narration.
- Maintain the speaker’s tone and emotions.
- If there is quoted dialogue within inverted commas, format it as (age-based gender: dialogue). Also, it should start from a new line. E.g., "\n\n(child female: The sky is way too beautiful)".
- Ensure smooth transitions between narration and background audio cues.
- Output only the required audiobook text, ensuring it is ready for direct use in a text-to-audio model.

**Format:**
- Keep narration natural and expressive, ensuring it feels like a true audiobook experience.
- If applicable, subtly describe non-verbal actions using immersive language.

**Output:**"""
    
    ab_data[p] = OpenAI_model_response(prompt, "gpt-4o")

# saving video audiobook (text-format)
# Ensure the directory exists
os.makedirs("Outputs", exist_ok=True)
with open("Outputs/audiobook_text_data.json", "w", encoding="utf-8") as json_file:
    json.dump(ab_data, json_file, indent=4)