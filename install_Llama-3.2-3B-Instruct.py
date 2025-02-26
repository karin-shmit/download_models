from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Set your Hugging Face token
huggingface_token = "your_huggingface_token_here"

# Set the environment variable for Hugging Face authentication
os.environ["HF_TOKEN"] = huggingface_token

# Specify the model name from Hugging Face
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Define the local directory to save the model
local_dir = f"./models/{model_name}"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Authenticate using the Hugging Face token
huggingface_token = "your_huggingface_token_here"
os.environ["HF_TOKEN"] = huggingface_token

# Download and save the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Automatically selects the appropriate dtype for the device
    device_map="auto",   # Automatically maps the model layers across available devices
    use_auth_token=huggingface_token
)

# Save the model and tokenizer locally
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print(f"Model and tokenizer saved to {local_dir}")
