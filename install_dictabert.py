from transformers import AutoModelForMaskedLM, AutoTokenizer
import os

# Specify the model name from Hugging Face
model_name = "dicta-il/dictabert"

# Define the local directory to save the model
local_dir = f"./models/{model_name}"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Download and save the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Save the model and tokenizer locally
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print(f"Model and tokenizer saved to {local_dir}")