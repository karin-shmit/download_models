from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Specify the model name from Hugging Face
model_name = "Qwen/Qwen2.5-3B-Instruct"

# Define the local directory to save the model
local_dir = f"./models/{model_name}"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Download and save the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Automatically selects the appropriate dtype for the device
    device_map="auto"   # Automatically maps the model layers across available devices
)

# Save the model and tokenizer locally
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print(f"Model and tokenizer saved to {local_dir}")