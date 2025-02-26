from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Specify the model name from Hugging Face
model_name = "dicta-il/dictalm2.0-instruct"

# Define the local directory to save the model
local_dir = f"./models/{model_name}"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Specify the device map (e.g., 'auto' or custom) and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download and save the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"  # This maps the model layers automatically across available devices
)

# Save the model and tokenizer locally
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print(f"Model and tokenizer saved to {local_dir}")


--------------------------------------


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Specify the local directory where the model is saved
local_dir = f"./models/{model_name}"

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_dir, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"  # Ensure this matches your device setup
)

messages = [
    {"role": "user", "content": "איזה רוטב אהוב עליך?"},
    {"role": "assistant", "content": "טוב, אני די מחבב כמה טיפות מיץ לימון סחוט טרי. זה מוסיף בדיוק את הכמות הנכונה של טעם חמצמץ לכל מה שאני מבשל במטבח!"},
    {"role": "user", "content": "האם יש לך מתכונים למיונז?"}
]

encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

generated_ids = model.generate(encoded, max_new_tokens=50, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])