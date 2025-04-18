import os
import requests
import json
import time

# Function to read prompts from prompts_obj.json
def read_prompts_from_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Read the prompts from prompts_obj.json
prompts_data = read_prompts_from_json("prompts_obj.json")

# Load definitions from definition.json
with open("definition.json", "r") as def_file:
    definitions = json.load(def_file)

# Function to submit a prompt to the imagine endpoint with retry logic
def submit_imagine(prompt, api_secret=os.environ.get("MIDJOURNEY_API_SECRET"), base64Array=None, notifyHook="", state=""):
    if base64Array is None:
        base64Array = []
    
    url = "https://lyu2401.zeabur.app/mj/submit/imagine"
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "mj-api-secret": api_secret
    }
    payload = {
        "base64Array": base64Array,
        "notifyHook": notifyHook,
        "prompt": prompt,
        "state": state
    }

    for attempt in range(3):  # Retry up to 3 times on error
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 502:
            print(f"502 Error: Retrying in 10 seconds... (Attempt {attempt + 1}/3)")
            time.sleep(10)  # Wait before retrying
        else:
            return {"error": f"Request failed with status code {response.status_code}", "details": response.text}
    
    return {"error": "Request failed after 3 retries", "details": response.text}

# Generate images for each prompt in the prompts_obj.json
for index, (prompt_name, prompt_text) in enumerate(prompts_data.items()):
    # Get the definition for the current category
    definition = definitions.get(prompt_name, "")

    # Modify each prompt by adding the definition
    high_prompt = f"{prompt_name} is {definition} {prompt_text[0]}"
    low_prompt = f"{prompt_name} is {definition} {prompt_text[1]}"
    
    print(f"Processing Prompt {index + 1}: {prompt_name}")
    
    for round_num in range(5):  # Five rounds per prompt
        for level, prompt in zip(['high', 'low'], [high_prompt, low_prompt]):
            result = submit_imagine(prompt)
            print(f"Prompt {index + 1} ({prompt_name}), Level {level}, Round {round_num + 1}: {result}")
            time.sleep(35)  # Delay to avoid rapid API calls
